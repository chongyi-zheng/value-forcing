import glob
import os
import re
import time
import types

import gymnasium
import numpy as np
import ogbench

from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Track episode rewards, lengths, and timing."""

    def __init__(self, env, filter_regexes=None):
        """Set up episode counters and optional info filters."""
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        """Reset the current episode's reward, length, and start time."""
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        """Step the environment, filter info, and record episode statistics."""
        observation, reward, terminated, truncated, info = self.env.step(action)

        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        """Reset episode statistics and the wrapped environment."""
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


def is_ocbench_env(env_name):
    """Check whether the name belongs to an OCBench environment family."""
    return env_name.split('-')[0] in ('block', 'chamber', 'switch', 'hanoi', 'bowling')


def make_env_and_datasets(
    env_name,
    *,
    action_clip_eps=1e-5,
    dataset_path=None,
    dataset_only=False,
    cur_env=None,
):
    """Load environments and datasets for training and evaluation.

    Dataset-only reloads need `cur_env` from an earlier call for relabeling or action normalization.
    OCBench also needs `dataset_path` and returns objects holding the normalizer; evaluation creates its environments.
    """

    if 'singletask' in env_name:
        dataset_kwargs = {}
        if dataset_path is not None:
            dataset_kwargs = dict(dataset_path=dataset_path, compact_dataset=False)
        if dataset_only:
            if cur_env is None:
                raise ValueError('cur_env must be provided when reloading single-task OGBench datasets.')
            env = eval_env = None
            train_dataset, val_dataset = ogbench.make_env_and_datasets(
                env_name,
                dataset_only=True,
                cur_env=cur_env,
                **dataset_kwargs,
            )
        else:
            env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, **dataset_kwargs)
            eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        if not dataset_only:
            env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
            eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset) if val_dataset is not None else None
    elif is_ocbench_env(env_name):
        try:
            from envs import ocbench_utils
        except ImportError as exc:
            raise ImportError(
                'OCBench dependencies are missing; install requirements.txt (see "Installation" in README.md).'
            ) from exc

        if env_name.split('-')[0] not in ('block', 'chamber', 'switch'):
            # Hanoi's failure terminations and bowling's pin rewards do not support the {-1, 0} reward shift.
            raise ValueError(f'Unsupported OCBench environment: {env_name}')
        if dataset_path is None:
            raise ValueError('OCBench environments need a dataset shard; pass --dataset_dir.')

        if dataset_only:
            if cur_env is None:
                raise ValueError('cur_env must be provided when reloading OCBench datasets.')
            env = eval_env = None
            action_normalizer = cur_env.action_normalizer
        else:
            # Use the first shard for consistent scaling across resumed runs, as in the OCBench reference.
            # Arm-action percentiles 1 and 99 map to -1 and 1; gripper actions stay in [-1, 1].
            first_path = sorted(
                path
                for path in glob.glob(os.path.join(os.path.dirname(dataset_path), '*.npz'))
                if not path.endswith('-val.npz')
            )[0]
            with np.load(first_path) as data:
                action_normalizer = ocbench_utils.ActionNormalizer.fit(data['actions'], percentile=1.0, clip=5.0)
            env = eval_env = types.SimpleNamespace(action_normalizer=action_normalizer)

        train_dataset = Dataset.create(**ocbench_utils.get_dataset(dataset_path, action_normalizer))
        val_path = dataset_path.removesuffix('.npz') + '-val.npz'
        val_dataset = None
        if os.path.exists(val_path):
            val_dataset = Dataset.create(**ocbench_utils.get_dataset(val_path, action_normalizer))

        # Keep dataset arm actions in [-5, 5]. Policies use [-1, 1], then evaluation reverses the scaling.
        return env, eval_env, train_dataset, val_dataset
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if not dataset_only:
        env.reset()
        eval_env.reset()

    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
