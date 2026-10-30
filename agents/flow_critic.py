from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ValueVectorField, ActorVectorField


class FlowCriticAgent(flax.struct.PyTreeNode):
    """FlowCritic at lambda=0 with best-of-N selection.

    https://arxiv.org/abs/2510.22686

    The critic learns distributional Bellman targets; candidates are ranked by truncated mean returns.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def aggregate_q(self, qs, config_key='q_agg'):
        """Aggregate critic members by minimum or mean."""
        if self.config[config_key] == 'min':
            return qs.min(axis=0)
        return qs.mean(axis=0)

    def flow_value_samples(
        self,
        observations,
        actions,
        rng,
        flow_network_name='critic_flow',
        num_value_samples=None,
        q_est=None,
    ):
        """Draw returns with Euler integration or estimate mean returns with a t=0 read.

        Aggregate ensemble members to shape (num_value_samples, *batch).
        """
        if num_value_samples is None:
            num_value_samples = self.config['num_value_samples']
        if q_est is None:
            q_est = 'euler'
        batch_shape = observations.shape[: -len(self.config['ob_dims'])]
        noises = jax.random.normal(rng, (num_value_samples, *batch_shape, 1))
        stacked = jnp.broadcast_to(noises, (self.config['num_critics'], *noises.shape))
        if q_est == 'flow_t0':
            returns = stacked + self.network.select(flow_network_name)(
                stacked, jnp.zeros_like(stacked), observations, actions
            )
        else:
            returns = self.compute_flow_returns(stacked, observations, actions, flow_network_name=flow_network_name)
        return self.aggregate_q(returns.squeeze(-1))

    def truncated_value(self, value_samples):
        """Average samples along the leading axis after dropping the largest `num_value_trunc`."""
        num_trunc = self.config['num_value_trunc']
        if num_trunc == 0:
            return value_samples.mean(axis=0)
        kept = jnp.sort(value_samples, axis=0)[: value_samples.shape[0] - num_trunc]
        return kept.mean(axis=0)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss and metrics."""
        batch_size = batch['action_chunks'].shape[0]
        rng, actor_rng, boot_rng, noise_rng, time_rng, diag_rng = jax.random.split(rng, 6)

        observations = batch['observations']
        action_chunks = batch['action_chunks']
        next_observations = batch['chunk_next_observations']
        chunk_valid = batch['chunk_valid']
        num_critics = self.config['num_critics']
        discount = self.config['discount'] ** self.config['action_chunk_size']

        next_actions, next_scores = self.sample_actions(
            next_observations,
            actor_rng,
            num_value_samples=self.config['bootstrap_num_value_samples'],
            q_est='flow_t0',
            return_scores=True,
        )

        # Ensemble members share bootstrap noise, independently of the interpolant noise.
        boot_noises = jax.random.normal(boot_rng, (batch_size, 1))
        boot_noises = jnp.broadcast_to(boot_noises, (num_critics, *boot_noises.shape))
        next_returns_e = self.compute_flow_returns(
            boot_noises, next_observations, next_actions, flow_network_name='target_critic_flow'
        )
        next_returns = self.aggregate_q(next_returns_e, config_key='ret_agg')
        returns = (
            jnp.expand_dims(batch['chunk_rewards'], axis=-1)
            + discount * jnp.expand_dims(batch['chunk_masks'], axis=-1) * next_returns
        )

        noises = jax.random.normal(noise_rng, (batch_size, 1))
        times = jax.random.uniform(time_rng, (batch_size, 1))
        noisy_returns = times * returns + (1 - times) * noises
        target_vector_field = returns - noises

        stacked_noisy_returns = jnp.broadcast_to(noisy_returns, (num_critics, *noisy_returns.shape))
        vector_fields = self.network.select('critic_flow')(
            stacked_noisy_returns, times, observations, action_chunks, params=grad_params
        )
        per_element_loss = (vector_fields - target_vector_field) ** 2
        cfm_loss = per_element_loss.sum(axis=0).mean(axis=-1)

        value_samples = self.flow_value_samples(
            observations, action_chunks, diag_rng, flow_network_name='target_critic_flow'
        )
        value_stds = value_samples.std(axis=0)

        critic_loss = cfm_loss
        critic_loss = critic_loss * chunk_valid
        critic_loss = critic_loss.mean()

        q = self.truncated_value(value_samples)

        info = {
            'critic_loss': critic_loss,
            'cfm_loss': cfm_loss.mean(),
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'value_std': value_stds.mean(),
            'bon_score_spread': (next_scores.max(axis=-1) - next_scores.mean(axis=-1)).mean(),
            'bon_argmax_frac0': (jnp.argmax(next_scores, axis=-1) == 0).mean(),
        }
        info['chunk_valid_frac'] = chunk_valid.mean()

        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss and metrics."""
        batch_size, ac_action_dim = batch['action_chunks'].shape
        chunk_valid = batch['chunk_valid']
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, ac_action_dim))
        x_1 = batch['action_chunks']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = (pred - vel) ** 2
        bc_flow_loss = bc_flow_loss * chunk_valid[:, None]
        bc_flow_loss = jnp.mean(bc_flow_loss)

        actor_loss = bc_flow_loss

        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['action_chunks']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the combined training loss and component metrics."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
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
        self.target_update(new_network, 'critic_flow')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('flow_network_name',))
    def compute_flow_returns(self, noises, observations, actions, flow_network_name='critic_flow'):
        """Integrate the return flow from t=0 to t=1, preserving the leading ensemble axis."""
        num_flow_steps = self.config['critic_num_flow_steps']
        noisy_returns = noises
        init_times = jnp.zeros((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        step_size = 1.0 / num_flow_steps

        def func(carry, i):
            """Advance the flow by one Euler step."""
            (noisy_returns,) = carry
            times = i * step_size + init_times
            vector_field = self.network.select(flow_network_name)(noisy_returns, times, observations, actions)
            new_noisy_returns = noisy_returns + step_size * vector_field
            return (new_noisy_returns,), None

        (noisy_returns,), _ = jax.lax.scan(func, (noisy_returns,), jnp.arange(num_flow_steps))

        return noisy_returns

    @jax.jit
    def compute_flow_actions(self, noises, observations):
        """Compute actions from the BC flow model using the Euler method."""
        num_flow_steps = self.config['num_flow_steps']
        noisy_actions = noises
        init_times = jnp.zeros((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        step_size = 1.0 / num_flow_steps

        def func(carry, i):
            """Advance the flow by one Euler step."""
            (noisy_actions,) = carry
            times = i * step_size + init_times
            vector_field = self.network.select('actor_flow')(observations, noisy_actions, times)
            new_noisy_actions = noisy_actions + vector_field * step_size
            return (new_noisy_actions,), None

        (noisy_actions,), _ = jax.lax.scan(func, (noisy_actions,), jnp.arange(num_flow_steps))

        noisy_actions = jnp.clip(noisy_actions, -1, 1)

        return noisy_actions

    @partial(jax.jit, static_argnames=('num_value_samples', 'q_est', 'return_scores'))
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
        *,
        num_value_samples=None,
        q_est=None,
        return_scores=False,
    ):
        """Sample actions from the policy."""
        num_samples = self.config['num_samples']
        action_seed, value_seed = jax.random.split(seed)

        actor_noises = jax.random.normal(
            action_seed,
            (*observations.shape[: -len(self.config['ob_dims'])], num_samples, self.config['ac_action_dim']),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, -2), num_samples, axis=-2)
        flow_actions = self.compute_flow_actions(actor_noises, n_observations)

        value_samples = self.flow_value_samples(
            n_observations,
            flow_actions,
            value_seed,
            num_value_samples=num_value_samples,
            q_est=q_est,
        )
        q = self.truncated_value(value_samples)

        if len(q.shape) > 1:
            actions = flow_actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            actions = flow_actions[jnp.argmax(q, axis=-1)]

        return (actions, q) if return_scores else actions

    @classmethod
    def create(cls, seed, example_batch, config):
        """Initialize the agent from an example batch and configuration."""
        assert 'action_chunks' in example_batch, (
            'flow_critic needs chunked batches: set dataset.action_chunk_size '
            "(utils.dataset_subsampling.prepare_dataset does this from config['action_chunk_size']) before sampling "
            'the example batch.'
        )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_action_chunks = example_batch['action_chunks']
        num_critics = config['num_critics']
        # Keep the leading ensemble axis even for one critic.
        ex_returns = ex_action_chunks[..., :1]
        ex_returns = jnp.broadcast_to(ex_returns, (num_critics, *ex_returns.shape))
        ex_times = ex_action_chunks[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        ac_action_dim = ex_action_chunks.shape[-1]
        assert ac_action_dim == config['action_chunk_size'] * action_dim, (
            f'action_chunks dim {ac_action_dim} != '
            f'action_chunk_size {config["action_chunk_size"]} * action_dim {action_dim}'
        )

        critic_flow_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=num_critics,
        )
        target_critic_flow_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=num_critics,
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        network_info = dict(
            critic_flow=(critic_flow_def, (ex_returns, ex_times, ex_observations, ex_action_chunks)),
            target_critic_flow=(target_critic_flow_def, (ex_returns, ex_times, ex_observations, ex_action_chunks)),
            actor_flow=(actor_flow_def, (ex_observations, ex_action_chunks, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic_flow'] = params['modules_critic_flow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['ac_action_dim'] = ac_action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the default agent configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='flow_critic',  # Agent registry key.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set automatically.
            action_dim=ml_collections.config_dict.placeholder(int),  # Set automatically.
            ac_action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            min_reward=ml_collections.config_dict.placeholder(float),  # Dataset metadata; unused by this agent.
            max_reward=ml_collections.config_dict.placeholder(float),  # Dataset metadata; unused by this agent.
            lr=3e-4,  # Optimizer learning rate.
            batch_size=1024,  # Training batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor hidden layer widths.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Critic flow hidden layer widths.
            actor_layer_norm=True,  # Actor layer normalization.
            value_layer_norm=True,  # Critic flow layer normalization.
            discount=0.999,  # Per environment step.
            tau=0.005,  # Target update rate.
            num_value_samples=10,  # Return samples per candidate score.
            num_value_trunc=1,  # Drop this many largest samples; must be below both value sample counts.
            num_critics=1,  # Critic flow ensemble size.
            ret_agg='mean',  # Target return ensemble aggregation: min or mean.
            q_agg='min',  # Scoring return ensemble aggregation: min or mean.
            num_samples=16,  # Candidates for best-of-N action selection.
            num_flow_steps=10,  # Actor flow integration steps.
            critic_num_flow_steps=10,  # Critic flow integration steps.
            bootstrap_num_value_samples=4,  # t=0 reads per backup score; None uses num_value_samples.
            log_grad_stats=False,  # Log gradient max, min, and norm.
            action_chunk_size=5,  # Prediction horizon.
            action_chunk_eval_size=ml_collections.config_dict.placeholder(int),  # Actions to execute; None uses all.
        )
    )
    return config
