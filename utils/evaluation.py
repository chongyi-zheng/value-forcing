import time
from collections import defaultdict, deque

import jax
import numpy as np
from tqdm import tqdm, trange


def supply_rng(f, rng):
    """Wrap a function to supply a fresh random seed on every call."""

    def wrapped(*args, **kwargs):
        """Split the stored key and call the function with the new seed."""
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten nested dictionaries by joining keys with a separator."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
):
    """Return evaluation statistics, trajectories, and videos from environment rollouts.

    Execute action chunks open-loop, optionally limiting their length with `action_chunk_eval_size`.
    Separate video episodes are rendered every `video_frame_skip` steps and excluded from statistics.
    """
    action_dim = None
    action_chunk_eval_size = None
    if config is not None and config.get('action_chunk_size', None) is not None:
        action_dim = config['action_dim']
        action_chunk_eval_size = config.get('action_chunk_eval_size', None)

    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        action_queue = deque()
        while not done:
            if len(action_queue) == 0:
                action = actor_fn(observations=observation, temperature=eval_temperature)
                action = np.array(action)
                action = np.clip(action, -1, 1)
                if action_dim is not None:
                    chunk = action.reshape(-1, action_dim)
                    if action_chunk_eval_size is not None:
                        chunk = chunk[:action_chunk_eval_size]
                    action_queue.extend(chunk)
                else:
                    action_queue.append(action)
            action = action_queue.popleft()

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def make_collection_seeds(root_seed, collection_idx, split_id, num_episodes):
    """Generate distinct episode reset seeds following the OCBench reference implementations.

    Dataset splits use IDs 0 to 3; evaluation uses split ID 10.
    """
    seed_sequence = np.random.SeedSequence([int(root_seed), int(collection_idx), int(split_id)])
    seeds = np.empty(num_episodes, dtype=np.uint32)
    root_mask = np.random.SeedSequence([int(root_seed)]).generate_state(1, dtype=np.uint32)[0]
    for idx, child in enumerate(seed_sequence.spawn(num_episodes)):
        packed = (int(collection_idx) << 20) | (int(split_id) << 18) | int(child.spawn_key[-1])
        # Mix the packed index into a 32-bit seed.
        value = (packed ^ int(root_mask)) & 0xFFFFFFFF
        value ^= value >> 16
        value = (value * 0x7FEB352D) & 0xFFFFFFFF
        value ^= value >> 15
        value = (value * 0x846CA68B) & 0xFFFFFFFF
        value ^= value >> 16
        seeds[idx] = value
    if len(np.unique(seeds)) != num_episodes:
        raise AssertionError('Duplicate collection seeds.')
    return seeds


def evaluate_mjwarp(
    agent,
    env_name,
    config=None,
    action_normalizer=None,
    num_eval_episodes=50,
    eval_temperature=0,
):
    """Evaluate parallel MJWarp worlds following OCBench, returning statistics and two empty lists.

    Execute chunks open-loop, with one world per episode. MJWarp does not truncate or reset episodes, so enforce
    time limits here and send zero actions to finished worlds. Trajectories and videos are omitted.
    """
    # Import MJWarp only when OCBench evaluation is requested.
    import ocbench
    from ocbench.mjwarp import make_env

    if num_eval_episodes == 0:
        return {}, [], []
    action_chunk_eval_size = None
    if config is not None and config.get('action_chunk_size', None) is not None:
        action_chunk_eval_size = config.get('action_chunk_eval_size', None)
    if action_chunk_eval_size is not None and action_chunk_eval_size < 1:
        raise ValueError('action_chunk_eval_size must be at least 1.')
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    family, backend, env_kwargs, max_episode_steps = ocbench.parse_env_spec(env_name)
    if backend != 'mjwarp':
        raise ValueError('MJWarp evaluation requires an environment name without `-cpu-`.')
    env = make_env(family, env_kwargs, num_eval_episodes)
    seeds = make_collection_seeds(np.random.randint(0, 2**32), 0, 10, num_eval_episodes)
    observations, _ = env.reset(seeds=seeds)

    action_dim = env.single_action_space.shape[-1]
    raw_action_low = env.single_action_space.low
    raw_action_high = env.single_action_space.high
    if action_normalizer is None:
        policy_action_low = raw_action_low
        policy_action_high = raw_action_high
    else:
        policy_action_low = action_normalizer.action_low
        policy_action_high = action_normalizer.action_high

    done = np.zeros(num_eval_episodes, dtype=bool)
    steps = np.zeros(num_eval_episodes, dtype=np.int32)
    returns = np.zeros(num_eval_episodes, dtype=np.float32)
    start_time = time.time()
    stats = defaultdict(list)

    with tqdm(total=num_eval_episodes, desc='eval', leave=False, dynamic_ncols=True) as pbar:
        while not np.all(done):
            actions = actor_fn(observations=observations, temperature=eval_temperature)
            # The agent emits one flattened chunk per world.
            action_chunks = np.asarray(actions, dtype=np.float32).reshape(num_eval_episodes, -1, action_dim)
            if action_chunk_eval_size is not None:
                action_chunks = action_chunks[:, :action_chunk_eval_size]
            action_chunks = np.clip(action_chunks, policy_action_low, policy_action_high)
            if action_normalizer is not None:
                action_chunks = action_normalizer.unnormalize(action_chunks)
            action_chunks = np.clip(action_chunks, raw_action_low, raw_action_high)

            for chunk_idx in range(action_chunks.shape[1]):
                active = ~done
                if not np.any(active):
                    break
                action = action_chunks[:, chunk_idx].copy()
                action[done] = 0.0
                observations, reward, terminated, _, info = env.step(action)
                steps[active] += 1
                returns[active] += reward[active]

                new_done = active & (terminated | (steps >= max_episode_steps))
                for world_id in np.where(new_done)[0]:
                    stats['success'].append(bool(info['success'][world_id]))
                    stats['failure'].append(bool(info['failure'][world_id]))
                    stats['healthy'].append(bool(info['healthy'][world_id]))
                    stats['episode.final_reward'].append(float(reward[world_id]))
                    stats['episode.return'].append(float(returns[world_id]))
                    stats['episode.length'].append(int(steps[world_id]))
                    stats['episode.duration'].append(time.time() - start_time)
                    pbar.update(1)
                done |= new_done

    # OCBench throughput is the success rate times the time limit divided by mean episode length.
    successes = np.asarray(stats['success'], dtype=np.float32)
    lengths = np.asarray(stats['episode.length'], dtype=np.float32)
    success_lengths = lengths[successes > 0]
    stats['mean_steps'].append(lengths.mean())
    stats['mean_success_steps'].append(success_lengths.mean() if len(success_lengths) else np.nan)
    stats['throughput'].append(successes.mean() * max_episode_steps / lengths.mean())
    stats['success_weighted_speed'].append(
        successes.mean() * max_episode_steps / success_lengths.mean() if len(success_lengths) else 0.0
    )

    for k, v in stats.items():
        stats[k] = np.mean(v)

    env.close()
    return stats, [], []
