import contextlib
import io
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import main


class ShardLifecycleTest(unittest.TestCase):
    def run_training(self, *, steps, interval, prefetch, start_step=0):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            dataset = mock.MagicMock()
            dataset.__contains__.return_value = True
            agent = mock.Mock()
            agent.update.return_value = (agent, {})
            agent_class = mock.Mock()
            agent_class.create.return_value = agent
            flags = SimpleNamespace(
                start_step=start_step,
                offline_steps=steps,
                restore_path=directory if start_step else None,
                restore_epoch=start_step if start_step else None,
                save_dir=directory,
                wandb_run_group='test',
                enable_wandb=0,
                seed=0,
                agent={'agent_name': 'test', 'batch_size': 2},
                dataset_dir='unused',
                num_datasets=None,
                dataset_replace_interval=interval,
                dataset_prefetch=prefetch,
                env_name='cube-single-play-singletask-task1-v0',
                video_episodes=0,
                reward_bounds_cache_dir=None,
                dataset_subset_size=None,
                dataset_subset_fraction=None,
                dataset_subset_temperature=1.0,
                log_interval=100,
                eval_interval=0,
                save_keep_last=0,
            )
            if start_step:
                Path(directory, f'params_{start_step}.pkl').touch()

            def submit(function, index):
                future = Future()
                future.set_result(function(index))
                return future

            pool = mock.Mock()
            pool.submit.side_effect = submit
            loader = stack.enter_context(
                mock.patch.object(main, 'make_env_and_datasets', return_value=(object(), object(), dataset, None))
            )
            executor = stack.enter_context(
                mock.patch.object(main.concurrent.futures, 'ThreadPoolExecutor', return_value=pool)
            )
            stack.enter_context(mock.patch.object(main, 'FLAGS', flags))
            stack.enter_context(mock.patch.object(main, 'get_flag_dict', return_value={}))
            stack.enter_context(mock.patch.object(main, 'get_exp_name', return_value='run'))
            stack.enter_context(mock.patch.object(main, 'get_dataset_paths', return_value=['a.npz', 'b.npz', 'c.npz']))
            stack.enter_context(mock.patch.object(main, 'read_reward_bounds', return_value=(-1.0, 0.0)))
            stack.enter_context(mock.patch.object(main, 'prepare_dataset', side_effect=lambda config, data, **kw: data))
            stack.enter_context(mock.patch.object(main, 'agents', {'test': agent_class}))
            stack.enter_context(mock.patch.object(main, 'restore_agent', return_value=agent))
            stack.enter_context(mock.patch.object(main.tqdm, 'tqdm', side_effect=lambda steps, **kw: steps))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main(None)
            self.assertEqual(agent.update.call_count, steps - start_step)
            return [call.kwargs['dataset_path'] for call in loader.call_args_list], executor, pool

    def test_final_update_does_not_load_unused_shard(self):
        paths, executor, _ = self.run_training(steps=4, interval=2, prefetch=0)
        self.assertEqual(paths, ['a.npz', 'b.npz'])
        executor.assert_not_called()

    def test_no_prefetch_when_first_swap_is_at_or_after_finish(self):
        for steps in (1, 2):
            with self.subTest(steps=steps):
                paths, executor, _ = self.run_training(steps=steps, interval=2, prefetch=1)
                self.assertEqual(paths, ['a.npz'])
                executor.assert_not_called()

    def test_resume_selects_current_shard_without_unused_prefetch(self):
        paths, executor, _ = self.run_training(steps=4, interval=2, prefetch=1, start_step=2)
        self.assertEqual(paths, ['b.npz'])
        executor.assert_not_called()

    def test_prefetch_only_loads_shards_used_by_remaining_updates(self):
        paths, executor, pool = self.run_training(steps=6, interval=2, prefetch=1)
        self.assertEqual(paths, ['a.npz', 'b.npz', 'c.npz'])
        executor.assert_called_once()
        self.assertEqual(pool.submit.call_count, 2)
        pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


if __name__ == '__main__':
    unittest.main()
