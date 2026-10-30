import numpy as np
import ocbench


class ActionNormalizer:
    """Scale actions using percentiles, following the OCBench reference."""

    def __init__(self, q_low, q_high, clip, gripper_dims=()):
        """Set scaling and clipping bounds, keeping gripper actions in [-1, 1]."""
        self.q_low = np.asarray(q_low, dtype=np.float32)
        self.q_high = np.asarray(q_high, dtype=np.float32)
        self.clip = float(clip)
        self.gripper_dims = np.zeros_like(self.q_low, dtype=bool)
        self.gripper_dims[np.asarray(gripper_dims, dtype=np.int64)] = True
        self.q_low[self.gripper_dims] = -1.0
        self.q_high[self.gripper_dims] = 1.0
        self.center = (self.q_low + self.q_high) / 2
        self.half_range = (self.q_high - self.q_low) / 2
        self.valid = self.half_range > 0
        self.action_low = np.full_like(self.q_low, -self.clip)
        self.action_high = np.full_like(self.q_high, self.clip)
        self.action_low[self.gripper_dims] = -1.0
        self.action_high[self.gripper_dims] = 1.0

    @classmethod
    def fit(cls, actions, percentile, clip):
        """Estimate action bounds from percentiles, treating the last dimension as the gripper."""
        q_low = np.percentile(actions, percentile, axis=0).astype(np.float32)
        q_high = np.percentile(actions, 100 - percentile, axis=0).astype(np.float32)
        return cls(q_low, q_high, clip, gripper_dims=(-1,))

    def normalize(self, actions):
        """Scale and clip actions, mapping constant dimensions to zero."""
        actions = np.asarray(actions, dtype=np.float32)
        normalized = np.zeros_like(actions, dtype=np.float32)
        normalized[..., self.valid] = (actions[..., self.valid] - self.center[self.valid]) / self.half_range[self.valid]
        return np.clip(normalized, self.action_low, self.action_high)

    def unnormalize(self, actions):
        """Clip normalized actions and restore their original scale."""
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, self.action_low, self.action_high)
        return actions * self.half_range + self.center


def get_dataset(dataset_path, action_normalizer):
    """Load OCBench transitions with normalized actions and rewards shifted from {0, 1} to {-1, 0}.

    `terminals` marks all episode ends, including time limits; `masks` is zero only at environment termination.
    """
    dataset = ocbench.load_dataset(dataset_path)
    if int(dataset['observation_interval']) != 1:
        raise ValueError(
            f'Only dense observations are supported; {dataset_path} stores one observation every '
            f'{int(dataset["observation_interval"])} steps.'
        )
    if np.any((dataset['masks'] == 0) != (dataset['rewards'] == 1)):
        # This reward shift requires episodes to terminate only on success.
        raise ValueError(f'{dataset_path} has an episode that terminates without success.')
    offsets, _ = ocbench.get_episode_offsets(dataset)
    # Each episode stores an extra final observation, so transition t in episode i uses row t + i.
    ob_idxs = np.arange(len(dataset['actions'])) + np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))

    return dict(
        observations=dataset['observations'][ob_idxs],
        actions=action_normalizer.normalize(dataset['actions']),
        rewards=dataset['rewards'] - 1.0,
        masks=dataset['masks'].astype(np.float32),
        terminals=dataset['terminals'].astype(np.float32),
        next_observations=dataset['observations'][ob_idxs + 1],
    )
