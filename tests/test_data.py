import unittest

import numpy as np

from utils.dataset_subsampling import prepare_dataset, subsample_dataset_trajectories
from utils.datasets import Dataset


def make_dataset(lengths):
    size = sum(lengths)
    terminals = np.zeros(size, dtype=np.float32)
    terminals[np.cumsum(lengths) - 1] = 1
    return Dataset.create(
        observations=np.arange(size, dtype=np.float32)[:, None],
        next_observations=np.arange(1, size + 1, dtype=np.float32)[:, None],
        actions=np.zeros((size, 1), dtype=np.float32),
        rewards=-np.ones(size, dtype=np.float32),
        masks=np.ones(size, dtype=np.float32),
        terminals=terminals,
    )


class TrajectorySubsetTest(unittest.TestCase):
    def test_full_fraction_accepts_one_trajectory(self):
        dataset = make_dataset([3])
        result = prepare_dataset({}, dataset, subsample=True, dataset_subset_fraction=1.0)
        np.testing.assert_array_equal(result['observations'], dataset['observations'])

    def test_fraction_never_rounds_above_budget(self):
        dataset = make_dataset([1, 1, 1])
        result = prepare_dataset({}, dataset, subsample=True, dataset_subset_fraction=0.5)
        self.assertEqual(result.size, 1)

    def test_nonpositive_budget_is_rejected(self):
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'must be positive'):
                prepare_dataset({}, make_dataset([1, 1]), subsample=True, dataset_subset_size=size)

    def test_fraction_too_small_for_one_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'must be positive'):
            prepare_dataset({}, make_dataset([1, 1, 1]), subsample=True, dataset_subset_fraction=0.1)

    def test_subset_keeps_whole_trajectories_in_original_order(self):
        dataset = make_dataset([2, 2, 2, 2])
        result = subsample_dataset_trajectories(dataset, size=5, seed=7)
        again = subsample_dataset_trajectories(dataset, size=5, seed=7)
        ids = result['observations'][:, 0].astype(int)
        self.assertEqual(len(ids), 4)
        np.testing.assert_array_equal(ids[1::2], ids[::2] + 1)
        self.assertTrue(np.all(np.diff(ids) > 0))
        np.testing.assert_array_equal(result['terminals'], [0, 1, 0, 1])
        np.testing.assert_array_equal(result['observations'], again['observations'])


if __name__ == '__main__':
    unittest.main()
