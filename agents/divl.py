import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, categorical_support


class DIVLAgent(flax.struct.PyTreeNode):
    """Distributional implicit value learning with best-of-N sampling.

    https://arxiv.org/abs/2605.00416

    The critic bootstraps from entropy-adaptive quantiles of a categorical value distribution.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def aggregate_q(self, qs):
        """Aggregate critic members by minimum or mean."""
        if self.config['q_agg'] == 'min':
            return qs.min(axis=0)
        return qs.mean(axis=0)

    def value_loss(self, batch, grad_params):
        """Compute the value loss and metrics."""
        target_qs = self.network.select('target_critic')(batch['observations'], actions=batch['action_chunks'])
        q = self.aggregate_q(target_qs)

        clipped_q = jnp.clip(q, self.config['v_min'], self.config['v_max'])
        target_bins = (clipped_q - self.config['v_min']) / self.config['delta_atom']
        lower_bins = jnp.floor(target_bins).astype(jnp.int32)
        upper_bins = jnp.ceil(target_bins).astype(jnp.int32)
        mass_lower = upper_bins + (lower_bins == upper_bins).astype(jnp.int32) - target_bins
        mass_upper = target_bins - lower_bins
        num_atoms = self.config['num_atoms']
        target_probs = (
            jax.nn.one_hot(lower_bins, num_atoms) * mass_lower[..., None]
            + jax.nn.one_hot(upper_bins, num_atoms) * mass_upper[..., None]
        )
        value_logits = self.network.select('value')(batch['observations'], params=grad_params)

        per_sample_value_loss = optax.softmax_cross_entropy(value_logits, target_probs)
        chunk_valid = batch['chunk_valid']
        per_sample_value_loss = per_sample_value_loss * chunk_valid
        value_loss = jnp.mean(per_sample_value_loss)

        probs = jax.nn.softmax(value_logits, axis=-1)
        v_mean = jnp.sum(probs * self.config['atoms'], axis=-1)
        info = {
            'value_loss': value_loss,
            'v_mean': v_mean.mean(),
            'v_max': v_mean.max(),
            'v_min': v_mean.min(),
            'q_target_mean': q.mean(),
            'target_clipped_frac': jnp.mean(
                ((q < self.config['v_min']) | (q > self.config['v_max'])).astype(jnp.float32)
            ),
        }
        info['chunk_valid_frac'] = chunk_valid.mean()
        return value_loss, info

    def critic_loss(self, batch, grad_params):
        """Compute the critic loss and metrics."""
        next_logits = self.network.select('value')(batch['chunk_next_observations'])
        next_probs = jax.nn.softmax(next_logits, axis=-1)

        num_atoms = self.config['num_atoms']
        entropy = -jnp.sum(next_probs * jnp.log(next_probs + 1e-8), axis=-1) / jnp.log(num_atoms)
        taus = self.config['tau_base'] - self.config['tau_alpha'] * entropy
        taus = jax.lax.stop_gradient(jnp.clip(taus, self.config['tau_min'], self.config['tau_max']))

        cdf = jnp.cumsum(next_probs, axis=-1)
        quantile_indices = jnp.sum((cdf < taus[..., None]).astype(jnp.int32), axis=-1)
        quantile_indices = jnp.clip(quantile_indices, 0, num_atoms - 1)
        next_v = self.config['atoms'][quantile_indices]

        discount = self.config['discount'] ** self.config['action_chunk_size']
        target_q = batch['chunk_rewards'] + discount * batch['chunk_masks'] * next_v
        target_q = jax.lax.stop_gradient(target_q)

        qs = self.network.select('critic')(batch['observations'], actions=batch['action_chunks'], params=grad_params)
        per_sample_critic_loss = jnp.sum((qs - target_q[None]) ** 2, axis=0)
        chunk_valid = batch['chunk_valid']
        per_sample_critic_loss = per_sample_critic_loss * chunk_valid
        critic_loss = jnp.mean(per_sample_critic_loss)

        q = self.aggregate_q(qs)
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            'next_v_mean': next_v.mean(),
            'tau_mean': taus.mean(),
            'tau_min': taus.min(),
            'tau_max': taus.max(),
            'value_entropy_mean': entropy.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss and metrics."""
        batch_size, ac_action_dim = batch['action_chunks'].shape
        x_rng, t_rng = jax.random.split(rng)

        x_0 = jax.random.normal(x_rng, (batch_size, ac_action_dim))
        x_1 = batch['action_chunks']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        per_sample_actor_loss = jnp.mean((pred - vel) ** 2, axis=-1)
        chunk_valid = batch['chunk_valid']
        per_sample_actor_loss = per_sample_actor_loss * chunk_valid
        actor_loss = jnp.mean(per_sample_actor_loss)

        return actor_loss, {'actor_loss': actor_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the combined training loss and component metrics."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

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
    def compute_flow_actions(self, noises, observations):
        """Compute actions from the BC flow model using the Euler method."""
        noisy_actions = noises
        step_size = 1.0 / self.config['num_flow_steps']

        def func(carry, i):
            """Advance the flow by one Euler step."""
            (noisy_actions,) = carry
            times = jnp.full((*noisy_actions.shape[:-1], 1), i * step_size)
            vector_field = self.network.select('actor_flow')(observations, noisy_actions, times)
            new_noisy_actions = noisy_actions + vector_field * step_size
            return (new_noisy_actions,), None

        (noisy_actions,), _ = jax.lax.scan(func, (noisy_actions,), jnp.arange(self.config['num_flow_steps']))

        noisy_actions = jnp.clip(noisy_actions, -1, 1)

        return noisy_actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0
    ):
        """Sample actions from the policy."""
        num_samples = self.config['num_samples']
        n_noises = jax.random.normal(
            seed, (*observations.shape[: -len(self.config['ob_dims'])], num_samples, self.config['ac_action_dim'])
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, -2), num_samples, axis=-2)
        n_actions = self.compute_flow_actions(n_noises, n_observations)

        qs = self.network.select('critic')(n_observations, actions=n_actions)
        q = self.aggregate_q(qs)
        if len(q.shape) > 1:
            return n_actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        return n_actions[jnp.argmax(q, axis=-1)]

    @classmethod
    def create(cls, seed, example_batch, config):
        """Initialize the agent from an example batch and configuration."""
        v_min, v_max, atoms, delta_atom = categorical_support(
            config['min_reward'], config['max_reward'], config['discount'], config['num_atoms']
        )
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        assert 'action_chunks' in example_batch, (
            'divl needs chunked batches: set dataset.action_chunk_size '
            "(utils.dataset_subsampling.prepare_dataset does this from config['action_chunk_size']) before sampling "
            'the example batch.'
        )

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_action_chunks = example_batch['action_chunks']
        ex_times = ex_action_chunks[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        ac_action_dim = ex_action_chunks.shape[-1]
        assert ac_action_dim == config['action_chunk_size'] * action_dim, (
            f'action_chunks dim {ac_action_dim} != '
            f'action_chunk_size {config["action_chunk_size"]} * action_dim {action_dim}'
        )

        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            value_dim=config['num_atoms'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            value_dim=1,
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
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

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['ac_action_dim'] = ac_action_dim
        # Return support uses the per-step discount.
        config['v_min'] = v_min
        config['v_max'] = v_max
        config['atoms'] = atoms
        config['delta_atom'] = delta_atom

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the default agent configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='divl',  # Agent registry key.
            min_reward=ml_collections.config_dict.placeholder(float),  # Set from the dataset.
            max_reward=ml_collections.config_dict.placeholder(float),  # Set from the dataset.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Inferred from the example batch.
            action_dim=ml_collections.config_dict.placeholder(int),  # Inferred from the example batch.
            ac_action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            lr=3e-4,  # Optimizer learning rate.
            batch_size=1024,  # Training batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor hidden layer widths.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Shared by the value and critic networks.
            value_layer_norm=True,  # Value and critic layer normalization.
            actor_layer_norm=True,  # Actor layer normalization.
            discount=0.999,  # Per environment step.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Critic aggregation for value targets and action ranking: 'min' or 'mean'.
            num_atoms=2001,  # Categorical value support size.
            v_min=ml_collections.config_dict.placeholder(float),  # Derived from reward bounds.
            v_max=ml_collections.config_dict.placeholder(float),  # Derived from reward bounds.
            atoms=ml_collections.config_dict.placeholder(jnp.ndarray),  # Derived atom locations.
            delta_atom=ml_collections.config_dict.placeholder(float),  # Derived atom spacing.
            tau_base=0.9,  # Bootstrap quantile level at zero entropy.
            tau_alpha=0.3,  # Quantile decrease per unit of normalized entropy.
            tau_min=0.0,  # Minimum quantile level.
            tau_max=1.0,  # Maximum quantile level.
            num_flow_steps=10,  # Actor Euler steps.
            num_samples=16,  # Best-of-N action candidates.
            action_chunk_size=5,  # Prediction horizon.
            action_chunk_eval_size=ml_collections.config_dict.placeholder(int),  # Actions to execute; None uses all.
            log_grad_stats=False,  # Log grad/max, grad/min, and grad/norm.
        )
    )
    return config
