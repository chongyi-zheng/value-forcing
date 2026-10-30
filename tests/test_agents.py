"""Regression checks for categorical supports and independently sized FloQ critics."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from agents import c51, divl, floq, implicit_c51
from utils.networks import categorical_support


class CategoricalSupportTest(unittest.TestCase):
    def test_standard_support_is_preserved(self):
        v_min, v_max, atoms, delta = categorical_support(-1.0, 0.0, 0.9, 5)
        self.assertAlmostEqual(v_min, -10.0)
        self.assertEqual(v_max, 0.0)
        self.assertAlmostEqual(delta, 2.5)
        np.testing.assert_allclose(atoms, np.linspace(-10.0, 0.0, 5))

    def test_invalid_supports_fail_before_network_initialization(self):
        for module, agent_class in (
            (c51, c51.C51Agent),
            (implicit_c51, implicit_c51.ImplicitC51Agent),
            (divl, divl.DIVLAgent),
        ):
            for min_reward, max_reward in ((0.0, 0.0), (-1.0, -1.0), (1.0, -1.0), (0.0, float('inf'))):
                with self.subTest(agent=module.__name__, bounds=(min_reward, max_reward)):
                    config = module.get_config()
                    config['min_reward'] = min_reward
                    config['max_reward'] = max_reward
                    with self.assertRaisesRegex(ValueError, 'min_reward < max_reward'):
                        agent_class.create(0, {}, config)

    def test_invalid_atom_count_and_discount(self):
        with self.assertRaisesRegex(ValueError, 'at least 2 atoms'):
            categorical_support(-1.0, 0.0, 0.9, 1)
        for discount in (-0.1, 1.0, float('nan')):
            with self.subTest(discount=discount), self.assertRaisesRegex(ValueError, 'discount'):
                categorical_support(-1.0, 0.0, discount, 5)


class FloQEnsembleTest(unittest.TestCase):
    def test_distinct_ensemble_sizes_train(self):
        batch = {
            'observations': jnp.ones((2, 3)),
            'actions': jnp.ones((2, 1)),
            'action_chunks': jnp.ones((2, 1)),
            'chunk_next_observations': jnp.ones((2, 3)),
            'chunk_rewards': jnp.zeros(2),
            'chunk_masks': jnp.ones(2),
            'chunk_valid': jnp.ones(2),
        }
        config = floq.get_config()
        config.update(
            actor_hidden_dims=(4, 4),
            value_hidden_dims=(4, 4),
            action_chunk_size=1,
            actor_flow_steps=1,
            critic_flow_steps=1,
            flow_num_ensembles=2,
            distill_num_ensembles=3,
            noise_samples=2,
            block_width=4,
            block_depth=1,
            min_reward=-1.0,
            max_reward=0.0,
        )
        agent = floq.FloQAgent.create(0, batch, config)
        updated, info = agent.update(batch)
        np.testing.assert_allclose(
            info['critic/critic_loss'],
            info['critic/floq_loss'] + info['critic/distilled_critic_loss'],
        )
        for parameter in jax.tree_util.tree_leaves(updated.network.params):
            self.assertTrue(np.isfinite(parameter).all())


if __name__ == '__main__':
    unittest.main()
