import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the largest leading dimension among dataset fields."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Dataset fields with transition and action-chunk sampling."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset, optionally making its arrays read-only."""
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        """Initialize dataset size and optional action-chunk settings."""
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.action_chunk_size = None  # Chunk length H; set externally.
        self.discount = None  # Per-step discount; set externally.

    def get_random_idxs(self, num_idxs):
        """Sample transition or chunk-start indices within array bounds, allowing trajectory crossings."""
        if self.action_chunk_size is not None:
            return np.random.randint(self.size - self.action_chunk_size + 1, size=num_idxs)
        return np.random.randint(self.size, size=num_idxs)

    def _add_action_chunks(self, batch, idxs):
        """Add flattened action chunks and their discounted H-step backup quantities to `batch`.

        Rewards stop after the first zero mask; backup quantities stop at trajectory terminals, including timeouts.
        Chunks crossing a trajectory end have `chunk_valid=0`, which every consuming loss must use as a weight.
        Boundary handling follows QGF's `sample_sequence`.
        """
        assert self.discount is not None, 'Set discount together with action_chunk_size.'
        chunk_size = self.action_chunk_size
        idxs = np.asarray(idxs)
        chunk_idxs = idxs[:, None] + np.arange(chunk_size)  # (B, H)
        actions = self._dict['actions'][chunk_idxs]  # (B, H, action_dim)
        batch['action_chunks'] = actions.reshape(len(idxs), -1)

        rewards = self._dict['rewards']
        discounts = self.discount ** np.arange(chunk_size)  # (H,)

        # Survival prefix excludes rewards after an absorbing step.
        masks = self._dict['masks'][chunk_idxs]  # (B, H)
        alive = np.ones_like(masks)
        alive[:, 1:] = np.cumprod(masks[:, :-1], axis=1)

        # valid[:, i] is 1 while no trajectory ends before step i.
        terminals = self._dict['terminals'][chunk_idxs]  # (B, H)
        term_cummax = np.maximum.accumulate(terminals, axis=1)
        valid = np.ones((len(idxs), chunk_size), dtype=np.float32)
        valid[:, 1:] = 1.0 - (term_cummax[:, :-1] > 0)

        batch['chunk_rewards'] = (rewards[chunk_idxs] * discounts * valid * alive).sum(axis=1).astype(rewards.dtype)

        # Minimum equals product for binary masks; ignore steps after the trajectory end.
        batch['chunk_masks'] = np.where(valid > 0, masks, np.inf).min(axis=1).astype(masks.dtype)

        last_valid = valid.sum(axis=1).astype(np.int64) - 1  # (B,)
        next_obs_idxs = idxs + last_valid
        batch['chunk_next_observations'] = jax.tree_util.tree_map(
            lambda arr: arr[next_obs_idxs], self._dict['next_observations']
        )

        batch['chunk_valid'] = valid[:, -1].astype(rewards.dtype)

    def sample(self, batch_size: int, idxs=None):
        """Sample transitions at random or supplied indices, adding action chunks when configured."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.action_chunk_size is not None:
            self._add_action_chunks(batch, idxs)
        return batch

    def get_subset(self, idxs):
        """Return dataset fields at the supplied indices."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
