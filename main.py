import concurrent.futures
import glob
import json
import os
import random
import time

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import is_ocbench_env, make_env_and_datasets
from utils.dataset_subsampling import prepare_dataset, read_reward_bounds
from utils.evaluation import evaluate, evaluate_mjwarp
from utils.flax_utils import prune_checkpoints, restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'value_forcing', 'Run group.')
flags.DEFINE_string('wandb_mode', 'online', 'Wandb mode.')
flags.DEFINE_integer('seed', 100, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task1-v0', 'Environment (task) name.')
flags.DEFINE_string(
    'dataset_dir', None, 'Directory of dataset shards (*.npz, each with its *-val.npz) or a single .npz file.'
)
flags.DEFINE_integer(
    'dataset_replace_interval', 0, 'Interval for cycling external dataset shards; 0 disables.', lower_bound=0
)
flags.DEFINE_integer(
    'dataset_prefetch',
    1,
    'Whether to load the next shard on a background thread while training continues.',
)
flags.DEFINE_integer('num_datasets', None, 'Maximum number of dataset shards to use.', lower_bound=1)
flags.DEFINE_integer(
    'dataset_subset_size',
    None,
    'If set, train on a reward-weighted random subset of whole trajectories holding '
    'at most this many training transitions. Mutually exclusive with '
    '--dataset_subset_fraction. With --dataset_replace_interval it is applied PER SHARD.',
)
flags.DEFINE_float(
    'dataset_subset_fraction',
    None,
    'If set, train on a reward-weighted random subset of whole trajectories holding '
    'at most this FRACTION (0 < f <= 1) of the loaded training transitions. Mutually '
    'exclusive with --dataset_subset_size. With --dataset_replace_interval it is '
    'applied PER SHARD.',
)
flags.DEFINE_float(
    'dataset_subset_temperature',
    1.0,
    'Softmax temperature for reward-weighted subsampling. Lower values favor '
    'high-reward trajectories more sharply; higher values are closer to uniform '
    'sampling.',
)
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Checkpoint directory; also the output directory when --start_step > 0.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_integer(
    'start_step',
    0,
    'Resume: first gradient step to run is start_step+1, so the steps already '
    'covered by the restored checkpoint are skipped. 0 trains from scratch. Set '
    'this together with --restore_path/--restore_epoch=<same step>.',
)
flags.DEFINE_integer(
    'save_keep_last',
    1,
    'Number of most recent checkpoints to keep after each save; for positive '
    'values only the final one is kept after training. 0 disables checkpoint '
    'saving, including the final checkpoint; -1 keeps every checkpoint.',
    lower_bound=-1,
)

flags.DEFINE_integer('offline_steps', 500000, 'Number of offline steps.', lower_bound=0)
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.', lower_bound=1)
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.', lower_bound=0)
flags.DEFINE_integer('save_interval', 50000, 'Saving interval.', lower_bound=1)
flags.DEFINE_string(
    'reward_bounds_cache_dir',
    None,
    'Directory for the cached reward bounds of a set of datasets.',
)

flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.', lower_bound=0)
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.', lower_bound=0)
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.', lower_bound=1)

config_flags.DEFINE_config_file('agent', 'agents/implicit_value_forcing.py', lock_config=False)
flags.mark_flag_as_required('dataset_dir')


def get_dataset_paths(dataset_dir):
    """Return training shard paths, excluding validation files when listing a directory."""
    dataset_dir = os.path.expanduser(dataset_dir)
    if os.path.isdir(dataset_dir):
        dataset_paths = sorted(glob.glob(os.path.join(dataset_dir, '*.npz')))
        dataset_paths = [path for path in dataset_paths if not os.path.basename(path).endswith('-val.npz')]
        if not dataset_paths:
            raise FileNotFoundError(f'No train .npz files found in dataset_dir={dataset_dir}')
        return dataset_paths

    return [dataset_dir]


def main(_):
    """Train, evaluate, and checkpoint an offline agent with optional dataset shard cycling."""
    if FLAGS.start_step < 0 or FLAGS.start_step > FLAGS.offline_steps:
        raise ValueError(f'--start_step={FLAGS.start_step} is outside [0, --offline_steps={FLAGS.offline_steps}].')
    if FLAGS.start_step > 0:
        if FLAGS.restore_path is None:
            raise ValueError(
                f'--start_step={FLAGS.start_step} skips the first '
                f'{FLAGS.start_step} gradient steps but no --restore_path was '
                f'given, so training would continue from RANDOM init. Pass the '
                f'checkpoint directory to resume from.'
            )
        if FLAGS.restore_epoch != FLAGS.start_step:
            raise ValueError(
                f'--restore_epoch={FLAGS.restore_epoch} != '
                f'--start_step={FLAGS.start_step}. Resuming from a checkpoint '
                f'taken at a different step than where the loop restarts would '
                f'silently train for the wrong number of updates.'
            )
    resuming = FLAGS.start_step > 0
    if resuming:
        restore_path = os.path.expanduser(FLAGS.restore_path)
        restore_dirs = [restore_path] if os.path.isdir(restore_path) else glob.glob(restore_path)
        if len(restore_dirs) != 1 or not os.path.isdir(restore_dirs[0]):
            raise ValueError('--restore_path must match exactly one checkpoint directory.')
        FLAGS.save_dir = os.path.realpath(restore_dirs[0])
        checkpoint_path = os.path.join(FLAGS.save_dir, f'params_{FLAGS.restore_epoch}.pkl')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        if FLAGS.start_step == FLAGS.offline_steps:
            print(f'[resume] checkpoint already reached step {FLAGS.offline_steps}; nothing to do.')
            return
        exp_name = os.path.basename(FLAGS.save_dir)
    else:
        exp_name = get_exp_name(FLAGS.seed)
        FLAGS.save_dir = os.path.join(os.path.expanduser(FLAGS.save_dir), FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    wandb_start_step = 0
    if FLAGS.enable_wandb:
        wandb_run = setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project='value-forcing',
            group=FLAGS.wandb_run_group,
            name=exp_name,
            mode=FLAGS.wandb_mode,
            resume=resuming,
        )
        if resuming:
            # Ordinary W&B resume preserves server history and only accepts newer steps.
            wandb_start_step = wandb_run.step
            if wandb_start_step > FLAGS.start_step + 1:
                print(f'[wandb] keeping existing history; logging new metrics from step {wandb_start_step}.')
    flag_dict = get_flag_dict()
    flags_name = f'resume_flags_{FLAGS.start_step}_{time.time_ns()}.json' if resuming else 'flags.json'
    with open(os.path.join(FLAGS.save_dir, flags_name), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    dataset_paths = get_dataset_paths(FLAGS.dataset_dir)
    if FLAGS.num_datasets is not None:
        dataset_paths = dataset_paths[: FLAGS.num_datasets]
    if not dataset_paths:
        raise ValueError('No dataset paths selected; check --dataset_dir and --num_datasets.')
    # Resume on the shard for the next step; checkpoints precede the shard swap.
    dataset_idx = (
        (FLAGS.start_step // FLAGS.dataset_replace_interval) % len(dataset_paths)
        if FLAGS.dataset_replace_interval > 0
        else 0
    )
    dataset_path = dataset_paths[dataset_idx]
    if dataset_path is not None:
        print(f'Using dataset shard: {dataset_path}')
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        dataset_path=dataset_path,
    )
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    normalizer_path = os.path.join(FLAGS.save_dir, 'action_normalizer.npz')
    if is_ocbench_env(FLAGS.env_name) and (not resuming or not os.path.exists(normalizer_path)):
        # Save the action normalizer needed to evaluate checkpoints.
        action_normalizer = eval_env.action_normalizer
        np.savez(
            normalizer_path,
            q_low=action_normalizer.q_low,
            q_high=action_normalizer.q_high,
            clip=action_normalizer.clip,
            gripper_dims=np.flatnonzero(action_normalizer.gripper_dims),
        )

    # Fix return support across all selected shards using task-specific rewards.
    min_reward, max_reward = read_reward_bounds(
        env, FLAGS.env_name, dataset_paths, cache_dir=FLAGS.reward_bounds_cache_dir
    )
    print(f'[reward_bounds] over {len(dataset_paths)} shard(s): min={min_reward:g} max={max_reward:g}')

    # Offset resumed RNG seeds to avoid replaying the original batch stream.
    if FLAGS.start_step > 0:
        random.seed(FLAGS.seed + FLAGS.start_step)
        np.random.seed((FLAGS.seed + FLAGS.start_step) % (2**32))
    else:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    # Subsample training only; keep validation complete for comparable metrics.
    dataset_kwargs = dict(
        seed=FLAGS.seed,
        dataset_subset_size=FLAGS.dataset_subset_size,
        dataset_subset_fraction=FLAGS.dataset_subset_fraction,
        dataset_subset_temperature=FLAGS.dataset_subset_temperature,
    )
    train_dataset = prepare_dataset(config, train_dataset, subsample=True, **dataset_kwargs)
    val_dataset = prepare_dataset(config, val_dataset, **dataset_kwargs)

    example_batch = train_dataset.sample(1)

    assert 'rewards' in train_dataset
    config['min_reward'] = min_reward
    config['max_reward'] = max_reward
    assert min_reward <= max_reward
    # Persist runtime reward bounds so restored agents use the same return support.
    reward_bounds_path = os.path.join(FLAGS.save_dir, 'reward_bounds.json')
    if not resuming or not os.path.exists(reward_bounds_path):
        with open(reward_bounds_path, 'w') as f:
            json.dump(
                {
                    'num_shards': len(dataset_paths),
                    'min_reward': float(min_reward),
                    'max_reward': float(max_reward),
                },
                f,
            )

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    if FLAGS.restore_path is not None:
        restore_path = glob.escape(FLAGS.save_dir) if resuming else FLAGS.restore_path
        agent = restore_agent(agent, restore_path, FLAGS.restore_epoch)

    if resuming:
        # Preserve checkpoints from the abandoned continuation before writing new ones.
        later_checkpoints = []
        for checkpoint in glob.glob(os.path.join(glob.escape(FLAGS.save_dir), 'params_*.pkl')):
            step = os.path.basename(checkpoint)[len('params_') : -len('.pkl')]
            if step.isdigit() and int(step) > FLAGS.start_step:
                later_checkpoints.append(checkpoint)
        if later_checkpoints:
            backup_dir = os.path.join(FLAGS.save_dir, f'resume_checkpoints_{FLAGS.start_step}_{time.time_ns()}')
            os.makedirs(backup_dir)
            for checkpoint in later_checkpoints:
                os.replace(checkpoint, os.path.join(backup_dir, os.path.basename(checkpoint)))
            print(f'[resume] backed up later checkpoints to {backup_dir}')
        print(f'[resume] restored step {FLAGS.start_step}; running steps {FLAGS.start_step + 1}..{FLAGS.offline_steps}')

    checkpoint_dirs = [FLAGS.save_dir]
    resume_step = FLAGS.start_step if resuming else None
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'), resume_step=resume_step)
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'), resume_step=resume_step)
    elapsed = float((train_logger.last_row or {}).get('time/total_time') or 0)
    last_time = time.time()
    first_time = last_time - elapsed

    def load_shard(idx):
        """Load, relabel, and prepare one training/validation shard pair."""
        _, _, shard_train, shard_val = make_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[idx],
            dataset_only=True,
            cur_env=env,
        )
        return (
            prepare_dataset(config, shard_train, subsample=True, **dataset_kwargs),
            prepare_dataset(config, shard_val, **dataset_kwargs),
        )

    # Prefetch only tasks whose relabeling does not consume the global NumPy RNG.
    prefetch_off_reason = None
    if not FLAGS.dataset_prefetch:
        prefetch_off_reason = '--dataset_prefetch=0'
    elif FLAGS.dataset_replace_interval == 0 or len(dataset_paths) <= 1:
        prefetch_off_reason = 'no shard cycling'
    elif (
        FLAGS.start_step // FLAGS.dataset_replace_interval + 1
    ) * FLAGS.dataset_replace_interval >= FLAGS.offline_steps:
        prefetch_off_reason = 'no shard swap before training finishes'
    elif not (is_ocbench_env(FLAGS.env_name) or any(k in FLAGS.env_name for k in ('cube', 'scene', 'puzzle'))):
        prefetch_off_reason = (
            f'{FLAGS.env_name} is not a manipspace env, so relabeling consumes the '
            f'global numpy RNG that also draws batch indices'
        )
    prefetch_pool = None
    prefetch_idx, prefetch_future = None, None
    if prefetch_off_reason is None:
        prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='shard-prefetch')
        prefetch_idx = (dataset_idx + 1) % len(dataset_paths)
        prefetch_future = prefetch_pool.submit(load_shard, prefetch_idx)
        print(f'[prefetch] on: shard {prefetch_idx} loading in the background')
    else:
        print(f'[prefetch] off ({prefetch_off_reason}); shard swaps run inline')

    for i in tqdm.tqdm(range(FLAGS.start_step + 1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.enable_wandb and i >= wandb_start_step:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_metrics = {}
            if is_ocbench_env(FLAGS.env_name):
                eval_info, _, renders = evaluate_mjwarp(
                    agent=agent,
                    env_name=FLAGS.env_name,
                    config=config,
                    action_normalizer=eval_env.action_normalizer,
                    num_eval_episodes=FLAGS.eval_episodes,
                )
            else:
                eval_info, _, renders = evaluate(
                    agent=agent,
                    env=eval_env,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            if FLAGS.enable_wandb and i >= wandb_start_step:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        if FLAGS.save_keep_last != 0 and (i % FLAGS.save_interval == 0 or i == FLAGS.offline_steps):
            save_agent(agent, FLAGS.save_dir, i)
            prune_checkpoints(checkpoint_dirs, FLAGS.save_keep_last)

        # Switch to the next shard for the following step, using prefetched data
        # when available. Wrap around after the last shard.
        if (
            i < FLAGS.offline_steps
            and FLAGS.dataset_replace_interval != 0
            and i % FLAGS.dataset_replace_interval == 0
            and len(dataset_paths) > 1
        ):
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f'Using dataset shard: {dataset_paths[dataset_idx]}')
            if prefetch_future is not None and prefetch_idx == dataset_idx:
                train_dataset, val_dataset = prefetch_future.result()
            else:
                train_dataset, val_dataset = load_shard(dataset_idx)
            if prefetch_pool is not None and i + FLAGS.dataset_replace_interval < FLAGS.offline_steps:
                prefetch_idx = (dataset_idx + 1) % len(dataset_paths)
                prefetch_future = prefetch_pool.submit(load_shard, prefetch_idx)

    if prefetch_pool is not None:
        prefetch_pool.shutdown(wait=False, cancel_futures=True)
    train_logger.close()
    eval_logger.close()

    if FLAGS.save_keep_last > 0:
        prune_checkpoints(checkpoint_dirs, 1)


if __name__ == '__main__':
    app.run(main)
