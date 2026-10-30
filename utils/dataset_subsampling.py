import hashlib
import json
import os
import time

import jax
import numpy as np

from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, get_size


def set_dataset_configs(config, dataset):
    """Set the action horizon and discount, leaving None unchanged."""
    if dataset is None:
        return None
    if config.get('action_chunk_size', None) is not None:
        dataset.action_chunk_size = config['action_chunk_size']
        dataset.discount = config['discount']
    return dataset


def trajectory_bounds(fields, n):
    """Return each trajectory's first and last row, including the final unfinished trajectory."""
    terminals = np.asarray(fields['terminals']).reshape(n, -1)
    ends = np.nonzero(np.any(terminals > 0, axis=-1))[0]
    if len(ends) == 0 or ends[-1] != n - 1:
        ends = np.append(ends, n - 1)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return starts, ends


def subsample_dataset_trajectories(dataset, size, seed=0, temperature=1.0):
    """Sample whole trajectories, favoring higher average rewards.

    Use `seed` for repeatable draws and `temperature` to control the preference for higher rewards.
    Keep the original row order and stop before exceeding `size` rows.
    """
    if size <= 0:
        raise ValueError(f'Trajectory subset size must be positive, got {size}.')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f'temperature must be finite and positive, got {temperature}.')

    fields = dict(dataset)
    n = get_size(fields)
    if n == 0:
        raise ValueError('Cannot subsample an empty dataset.')
    if size >= n:
        print(f'[subsample/traj] requested {size} >= dataset size {n}; using the full dataset.')
        return fields
    starts, ends = trajectory_bounds(fields, n)
    lengths = ends - starts + 1
    subset_size = int(size)
    if subset_size < int(lengths.min()):
        raise ValueError(
            f'Trajectory-level subsampling target {subset_size} rows is smaller than the '
            f'shortest trajectory ({int(lengths.min())} rows), so no whole '
            f'trajectory fits. Raise --dataset_subset_fraction/_size.'
        )

    rewards = np.asarray(fields['rewards'], dtype=np.float64).reshape(n, -1).sum(axis=-1)
    if not np.all(np.isfinite(rewards)):
        raise ValueError('Cannot subsample with non-finite transition rewards.')

    traj_returns = np.add.reduceat(rewards, starts)
    traj_scores = traj_returns / lengths

    rng = np.random.default_rng(seed)
    gumbel = rng.gumbel(size=len(starts))
    logits = (traj_scores - traj_scores.max()) / temperature

    def greedy_prefix(scores):
        """Keep trajectories until the next exceeds the row budget; return them in dataset order."""
        order = np.argsort(-scores, kind='stable')
        kept = order[np.cumsum(lengths[order]) <= subset_size]
        return np.sort(kept), int(lengths[kept].sum())

    kept, total = greedy_prefix(logits + gumbel)
    if len(kept) == 0:
        raise ValueError(f'Trajectory-level subsampling kept 0 trajectories for subset size {subset_size}.')

    idxs = np.concatenate([np.arange(starts[t], ends[t] + 1) for t in kept])
    subset = jax.tree_util.tree_map(lambda arr: np.asarray(arr)[idxs], fields)

    # Check that gaps in the selected rows occur only between trajectories.
    is_discontinuous = np.zeros(len(idxs), dtype=bool)
    is_discontinuous[:-1] = idxs[1:] != idxs[:-1] + 1
    sub_terminals = np.asarray(subset['terminals']).reshape(len(idxs), -1)
    unmarked = int((is_discontinuous & ~np.any(sub_terminals > 0, axis=-1)).sum())
    assert unmarked == 0, (
        f'{unmarked} row(s) of the subset are followed by another trajectory but carry no '
        f'terminal marker; trajectory-level subsetting is broken.'
    )

    weights = np.exp(logits)
    weights = weights / weights.sum()
    print(
        f'[subsample/traj] kept {len(kept)}/{len(starts)} trajectories = '
        f'{total}/{n} training transitions ({total / n:.1%}) '
        f'subset_size={subset_size} ({total / max(subset_size, 1):.1%} of total) seed={seed}; '
        f'temperature={temperature:g}; '
        f'softmax_traj_weight range=[{weights.min():.3g}, {weights.max():.3g}], '
        f'chosen_mean_step_reward={traj_scores[kept].mean():.3g} '
        f'(all trajectories {traj_scores.mean():.3g}), '
        f'traj_len=[{int(lengths[kept].min())}, {int(lengths[kept].max())}], artificial_terminals=0.'
    )

    return subset


def read_reward_bounds(env, env_name, dataset_paths, cache_dir=None):
    """Find the minimum and maximum rewards across all training shards.

    Load rewards for the current task before subsampling. Cache results in an existing `cache_dir` or
    `$REWARD_BOUNDS_CACHE_DIR`, when set; clear the cache if reward processing changes.
    """
    cache_path = None
    cache_dir = cache_dir or os.environ.get('REWARD_BOUNDS_CACHE_DIR')
    if cache_dir:
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f'Reward-bounds cache directory not found: {cache_dir}')
        fingerprint = [env_name]
        for path in dataset_paths:
            if path is None:
                continue
            try:
                st = os.stat(path)
            except OSError:
                fingerprint = None
                break
            fingerprint.append(f'{os.path.basename(path)}:{st.st_size}:{st.st_mtime_ns}')
        if fingerprint is not None:
            key = hashlib.sha256('|'.join(fingerprint).encode()).hexdigest()[:32]
            cache_path = os.path.join(cache_dir, f'{key}.json')
            try:
                with open(cache_path) as f:
                    hit = json.load(f)
                print(f'[reward_bounds] cache HIT {cache_path}: min={hit["min"]:g} max={hit["max"]:g}')
                return hit['min'], hit['max']
            except (OSError, ValueError, KeyError):
                print(f'[reward_bounds] cache miss; scanning {len(dataset_paths)} shard(s)')

    lo, hi = float('inf'), float('-inf')
    start = time.time()
    for idx, path in enumerate(dataset_paths):
        if path is None:
            continue
        _, _, shard, _ = make_env_and_datasets(
            env_name,
            dataset_path=path,
            dataset_only=True,
            cur_env=env,
        )
        rewards = np.asarray(shard['rewards'], dtype=np.float64)
        lo = min(lo, float(rewards.min()))
        hi = max(hi, float(rewards.max()))
        if (idx + 1) % 10 == 0 or idx + 1 == len(dataset_paths):
            print(
                f'[reward_bounds] scanned {idx + 1}/{len(dataset_paths)} shards, '
                f'running range [{lo:g}, {hi:g}] ({time.time() - start:.0f}s)'
            )
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(
            f'The reward-bounds scan found no rewards over {len(dataset_paths)} '
            f'shard path(s) for env_name={env_name!r}.'
        )

    if cache_path is not None:
        # Finish writing before making the cache entry available to other readers.
        try:
            tmp_path = f'{cache_path}.{os.getpid()}.tmp'
            with open(tmp_path, 'w') as f:
                json.dump({'min': lo, 'max': hi, 'env_name': env_name, 'num_shards': len(dataset_paths)}, f)
            os.replace(tmp_path, cache_path)
            print(f'[reward_bounds] cached to {cache_path}')
        except OSError as exc:
            print(f'[reward_bounds] could not cache to {cache_path}: {exc}')

    return lo, hi


def prepare_dataset(
    config,
    dataset,
    subsample=False,
    seed=0,
    dataset_subset_size=None,
    dataset_subset_fraction=None,
    dataset_subset_temperature=1.0,
):
    """Build a dataset with the agent's settings and an optional trajectory subset.

    Choose either a row count or a fraction for subsampling. Return None if no dataset is provided.
    """
    if dataset is None:
        return None
    fields = dataset

    if dataset_subset_size is not None and dataset_subset_fraction is not None:
        raise ValueError(
            f'--dataset_subset_size={dataset_subset_size} and --dataset_subset_fraction={dataset_subset_fraction} '
            f'are mutually exclusive; pass exactly one.'
        )
    subset_size = None
    if dataset_subset_fraction is not None:
        if not 0.0 < dataset_subset_fraction <= 1.0:
            raise ValueError(f'--dataset_subset_fraction must be in (0, 1], got {dataset_subset_fraction}.')
        # Convert to a dict so get_size counts data rows, not dataset fields.
        subset_size = int(dataset_subset_fraction * get_size(dict(fields)))
    elif dataset_subset_size is not None:
        subset_size = int(dataset_subset_size)

    if subsample and subset_size is not None:
        fields = subsample_dataset_trajectories(
            fields,
            size=subset_size,
            seed=seed,
            temperature=dataset_subset_temperature,
        )
    return set_dataset_configs(config, Dataset.create(**fields))
