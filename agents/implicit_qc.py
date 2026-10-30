import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP_CLASSES, ActorVectorField, Value


class ImplicitQCAgent(flax.struct.PyTreeNode):
    """Implicit QC with best-of-N selection.

    Uses expectile value regression, a twin critic, and a behavioral cloning flow.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Weight squared errors by the sign of the advantage."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the value loss and metrics."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], actions=batch['action_chunks'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], params=grad_params)
        per_sample_value_loss = self.expectile_loss(q - v, q - v, self.config['expectile'])
        chunk_valid = batch['chunk_valid']
        per_sample_value_loss = per_sample_value_loss * chunk_valid
        value_loss = per_sample_value_loss.mean()

        info = {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }
        info['chunk_valid_frac'] = chunk_valid.mean()
        return value_loss, info

    def critic_loss(self, batch, grad_params):
        """Compute the critic loss and metrics."""
        next_v = self.network.select('value')(batch['chunk_next_observations'])
        discount = self.config['discount'] ** self.config['action_chunk_size']
        q = batch['chunk_rewards'] + discount * batch['chunk_masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], actions=batch['action_chunks'], params=grad_params
        )
        per_sample_critic_loss = (q1 - q) ** 2 + (q2 - q) ** 2
        chunk_valid = batch['chunk_valid']
        per_sample_critic_loss = per_sample_critic_loss * chunk_valid
        critic_loss = per_sample_critic_loss.mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss and metrics."""
        batch_size, ac_action_dim = batch['action_chunks'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, ac_action_dim))
        x_1 = batch['action_chunks']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        per_sample_actor_loss = jnp.mean((pred - vel) ** 2, axis=-1)
        chunk_valid = batch['chunk_valid']
        per_sample_actor_loss = per_sample_actor_loss * chunk_valid
        actor_loss = per_sample_actor_loss.mean()

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the combined training loss and component metrics."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Soft-update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Return the updated agent and training metrics."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            """Evaluate the training loss for the candidate parameters."""
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn, log_grad_stats=self.config['log_grad_stats'])
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the policy."""
        seed, action_seed = jax.random.split(seed)

        n_actions = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-1],
                self.config['num_samples'],
                self.config['ac_action_dim'],
            ),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, -2), self.config['num_samples'], axis=-2)
        for i in range(self.config['flow_steps']):
            t = jnp.full((*n_actions.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(n_observations, n_actions, t)
            n_actions = n_actions + vels / self.config['flow_steps']
        n_actions = jnp.clip(n_actions, -1, 1)

        q = self.network.select('critic')(n_observations, actions=n_actions).min(axis=0)
        if len(q.shape) > 1:
            actions = n_actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            actions = n_actions[jnp.argmax(q, axis=-1)]
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Initialize the agent from an example batch and configuration."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        assert 'action_chunks' in example_batch, (
            'implicit_qc needs chunked batches: set dataset.action_chunk_size '
            "(utils.dataset_subsampling.prepare_dataset does this from config['action_chunk_size']) before sampling "
            'the example batch.'
        )

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_action_chunks = example_batch['action_chunks']
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        ac_action_dim = ex_action_chunks.shape[-1]
        assert ac_action_dim == config['action_chunk_size'] * action_dim, (
            f'action_chunks dim {ac_action_dim} != '
            f'action_chunk_size {config["action_chunk_size"]} * action_dim {action_dim}'
        )

        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        mlp_class = MLP_CLASSES[config['mlp_class']]

        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            mlp_class=mlp_class,
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            mlp_class=mlp_class,
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
            mlp_class=mlp_class,
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_action_chunks)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_action_chunks)),
            actor_flow=(actor_flow_def, (ex_observations, ex_action_chunks, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ex_observations.shape[1:]
        config['action_dim'] = action_dim
        config['ac_action_dim'] = ac_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the default agent configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='implicit_qc',  # Agent registry key.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set from the example batch.
            action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            ac_action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            lr=3e-4,  # Optimizer learning rate.
            batch_size=1024,  # Training batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor hidden layer widths.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value and critic networks.
            layer_norm=True,  # Value and critic layer normalization.
            actor_layer_norm=True,  # Actor layer normalization.
            mlp_class='mlp',  # 'mlp' or 'resmlp'.
            discount=0.999,  # Per environment step.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL value expectile.
            num_samples=16,  # Action candidates for best-of-N sampling.
            flow_steps=10,  # Actor Euler steps.
            action_chunk_size=5,  # Prediction horizon.
            action_chunk_eval_size=ml_collections.config_dict.placeholder(int),  # Actions to execute; None uses all.
            log_grad_stats=False,  # Log grad/max, grad/min, and grad/norm.
        )
    )
    return config
