import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, ImplicitQuantileValue


jax_float_dtype = jax.dtypes.canonicalize_dtype(jnp.float64)


class ImplicitIQNAgent(flax.struct.PyTreeNode):
    """IQN with expectile-weighted value quantiles and best-of-N flow sampling.

    https://arxiv.org/abs/1806.06923

    Critic backups use the next-state value.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def quantile_huber_loss(td_errors, taus, kappa):
        """Sum quantile Huber errors over online fractions and average over target fractions."""
        element_wise_huber_loss = jnp.where(
            jnp.abs(td_errors) <= kappa,
            0.5 * td_errors**2,
            kappa * (jnp.abs(td_errors) - 0.5 * kappa),
        )

        huber_loss = (
            jnp.abs(taus[..., None] - (td_errors < 0).astype(jax_float_dtype)) * element_wise_huber_loss / kappa
        )

        return huber_loss.sum(axis=-2).mean(axis=-1)

    def value_loss(self, batch, grad_params, rng):
        """Compute the value loss and metrics."""
        batch_size = batch['observations'].shape[0]
        rng, tau_rng = jax.random.split(rng, 2)

        taus = jax.random.uniform(tau_rng, shape=(batch_size, self.config['num_quantiles']))

        target_quantiles = self.network.select('target_critic')(batch['observations'], batch['action_chunks'], taus)
        target_qs = target_quantiles.squeeze(-1).mean(axis=-1)
        # For 'min', select one member by Q rather than minimizing each quantile.
        if self.config['quantile_agg'] == 'min':
            member_idxs = jnp.argmin(target_qs, axis=0)
            batch_idxs = jnp.arange(target_qs.shape[1])
            target_quantiles = target_quantiles[member_idxs, batch_idxs]
            q = target_qs[member_idxs, batch_idxs]
        else:
            target_quantiles = target_quantiles.mean(axis=0)
            q = target_qs.mean(axis=0)

        value_quantiles = self.network.select('value')(batch['observations'], taus=taus, params=grad_params)
        v = value_quantiles.squeeze(-1).mean(axis=-1)

        per_sample_mse = jnp.mean((target_quantiles - value_quantiles) ** 2, axis=(-2, -1))

        adv = q - v
        weights = jnp.where(adv >= 0, self.config['expectile'], 1 - self.config['expectile'])
        per_sample_value_loss = weights * per_sample_mse
        chunk_valid = batch['chunk_valid']
        per_sample_value_loss = per_sample_value_loss * chunk_valid
        value_loss = jnp.mean(per_sample_value_loss)

        frac_pos = (adv >= 0).mean()
        info = {
            'value_loss': value_loss,
            'value_mse_loss': per_sample_mse.mean(),
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'adv_mean': adv.mean(),
            'adv_frac_pos': frac_pos,
            'expectile_w_mean': weights.mean(),
        }
        info['chunk_valid_frac'] = chunk_valid.mean()
        return value_loss, info

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss and metrics."""
        batch_size = batch['observations'].shape[0]
        rng, tau_rng, next_tau_rng = jax.random.split(rng, 3)

        taus = jax.random.uniform(tau_rng, shape=(batch_size, self.config['num_quantiles']))
        quantiles = self.network.select('critic')(
            batch['observations'], batch['action_chunks'], taus, params=grad_params
        )  # (num_ensembles, batch_size, num_quantiles, 1)

        next_taus = jax.random.uniform(next_tau_rng, shape=(batch_size, self.config['num_quantiles']))
        next_value_quantiles = self.network.select('value')(batch['chunk_next_observations'], taus=next_taus)
        next_value_quantiles = next_value_quantiles.transpose([0, 2, 1])  # (batch_size, 1, num_quantiles)
        discount = self.config['discount'] ** self.config['action_chunk_size']
        target_quantiles = (
            batch['chunk_rewards'][:, None, None]
            + discount * batch['chunk_masks'][:, None, None] * next_value_quantiles
        )

        td_errors = target_quantiles[None] - quantiles  # (num_ensembles, batch_size, num_taus, num_tau_primes)
        per_sample_critic_loss = self.quantile_huber_loss(td_errors, taus, self.config['kappa'])
        chunk_valid = batch['chunk_valid']
        per_sample_critic_loss = per_sample_critic_loss * chunk_valid
        critic_loss = per_sample_critic_loss.mean()

        q = jnp.mean(quantiles.squeeze(-1), axis=-1)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
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

        info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the combined training loss and component metrics."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, value_rng, actor_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + value_loss + actor_loss
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
    def compute_flow_actions(
        self,
        noises,
        observations,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        noisy_actions = noises
        init_times = jnp.zeros((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        end_times = jnp.ones((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            """Advance the flow by one Euler step."""
            (noisy_actions,) = carry

            times = i * step_size + init_times
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
        temperature=1.0,
    ):
        """Sample actions from the policy."""
        noise_seed, tau_seed = jax.random.split(seed)
        n_noises = jax.random.normal(
            noise_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['num_samples'],
                self.config['ac_action_dim'],
            ),
        )
        n_observations = jnp.repeat(
            jnp.expand_dims(observations, -2),
            self.config['num_samples'],
            axis=-2,
        )
        n_actions = self.compute_flow_actions(n_noises, n_observations)

        taus = jax.random.uniform(tau_seed, shape=(self.config['num_samples'], self.config['num_quantiles']))
        quantiles = self.network.select('critic')(n_observations, n_actions, taus)
        if self.config['quantile_agg'] == 'min':
            quantiles = quantiles.min(axis=0)
        else:
            quantiles = quantiles.mean(axis=0)
        q = jnp.mean(quantiles.squeeze(-1), axis=-1)

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
        rng, tau_rng, init_rng = jax.random.split(rng, 3)

        assert 'action_chunks' in example_batch, (
            'implicit_iqn needs chunked batches: set dataset.action_chunk_size '
            "(utils.dataset_subsampling.prepare_dataset does this from config['action_chunk_size']) before sampling "
            'the example batch.'
        )

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_action_chunks = example_batch['action_chunks']
        ex_taus = jax.random.uniform(tau_rng, shape=(*ex_observations.shape[:-1], config['num_quantiles']))
        ex_times = ex_action_chunks[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        ac_action_dim = ex_action_chunks.shape[-1]
        assert ac_action_dim == config['action_chunk_size'] * action_dim, (
            f'action_chunks dim {ac_action_dim} != '
            f'action_chunk_size {config["action_chunk_size"]} * action_dim {action_dim}'
        )

        value_def = ImplicitQuantileValue(
            hidden_dims=config['value_hidden_dims'],
            tau_embedding_num_cosines=config['num_cosines'],
            embedding_dim=config['embedding_dim'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
        )
        critic_def = ImplicitQuantileValue(
            hidden_dims=config['value_hidden_dims'],
            tau_embedding_num_cosines=config['num_cosines'],
            embedding_dim=config['embedding_dim'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        network_info = dict(
            value=(value_def, dict(observations=ex_observations, taus=ex_taus)),
            critic=(critic_def, (ex_observations, ex_action_chunks, ex_taus)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_action_chunks, ex_taus)),
            actor_flow=(actor_flow_def, (ex_observations, ex_action_chunks, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        # Exclude the zero-gradient target critic from Adam moments; target_update handles its parameters.
        network_tx = optax.masked(
            network_tx,
            lambda params: {k: k != 'modules_target_critic' for k in params},
        )
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['ac_action_dim'] = ac_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the default agent configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='implicit_iqn',  # Agent registry key.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set from the example batch.
            action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            ac_action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            lr=3e-4,  # Optimizer learning rate.
            batch_size=1024,  # Training batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor hidden layer widths.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value and critic hidden layer widths.
            value_layer_norm=True,  # Value and critic layer normalization.
            actor_layer_norm=True,  # Actor layer normalization.
            num_cosines=64,  # Cosines in the quantile fraction embedding.
            embedding_dim=256,  # Input and quantile fraction embedding size.
            discount=0.999,  # Per environment step.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            kappa=300.0,  # Quantile Huber threshold; the paper uses 30 for humanoidmaze-giant.
            quantile_agg='mean',  # 'min' selects by Q for value targets and per quantile for sampling; or 'mean'.
            num_quantiles=8,  # Fractions sampled per network evaluation.
            num_samples=16,  # Action candidates for best-of-N sampling.
            num_flow_steps=10,  # Actor Euler steps.
            log_grad_stats=False,  # Log grad/max, grad/min, and grad/norm.
            action_chunk_size=5,  # Prediction horizon.
            action_chunk_eval_size=ml_collections.config_dict.placeholder(int),  # Actions to execute; None uses all.
        )
    )
    return config
