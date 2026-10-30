from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP_CLASSES, ValueVectorField, ActorVectorField, UnconditionalEmbedding


class IVFAgent(flax.struct.PyTreeNode):
    """Implicit value forcing with best-of-N selection.

    The critic and value share an ensemble flow: actions select Q, and learned null actions select V.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def ensemble_agg(x, mode):
        """Aggregate ensemble members by minimum or mean."""
        if mode == 'min':
            return jnp.min(x, axis=0)
        return jnp.mean(x, axis=0)

    def return_support(self):
        """Return scaled empirical bounds, which must match training at evaluation."""
        factor = 1.0 - self.config['discount'] ** self.config['action_chunk_size']
        return (
            factor * self.config['min_reward'] / (1 - self.config['discount']),
            factor * self.config['max_reward'] / (1 - self.config['discount']),
        )

    def clip_return_support(self, returns):
        """Clip returns to empirical bounds when enabled."""
        if not self.config['clip_returns']:
            return returns
        lo, hi = self.return_support()
        return jnp.clip(returns, lo, hi)

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Weight squared errors by the sign of the advantage."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def compute_qs(self, observations, actions, noises, module_name='critic_flow'):
        """Estimate Q by integrating to t=1, averaging return samples, and aggregating ensemble members."""
        num_ret_samples = self.config['num_ret_samples']
        n_observations = jnp.broadcast_to(observations, (num_ret_samples, *observations.shape))
        n_actions = jnp.broadcast_to(actions, (num_ret_samples, *actions.shape))

        qs = (
            self.compute_flow_returns(noises, n_observations, actions=n_actions, module_name=module_name)
            .squeeze(-1)
            .mean(axis=1)
        )

        return self.ensemble_agg(qs, self.config['q_agg'])

    def compute_qs_flow_t0(self, observations, rng, actions=None, module_name='critic_flow', num_ret_samples=None):
        """Estimate Q with z^0 + v(z^0, 0), without ODE integration.

        With `actions=None`, learned null actions select V. Ensemble members share the base noise.
        """
        if num_ret_samples is None:
            num_ret_samples = self.config['num_ret_samples']
        batch_shape = observations.shape[: -len(self.config['ob_dims'])]
        noises = jax.random.normal(rng, (num_ret_samples, *batch_shape, 1))
        n_observations = jnp.broadcast_to(observations, (num_ret_samples, *observations.shape))
        if actions is None:
            n_actions = jnp.expand_dims(self.network.select('cfg_uncond_embed')(), tuple(range(1, noises.ndim)))
        else:
            n_actions = jnp.broadcast_to(actions, (num_ret_samples, *actions.shape))
        times = jnp.zeros_like(noises)

        e_noises = jnp.broadcast_to(noises, (self.config['num_vels'], *noises.shape))
        vels = self.network.select(module_name)(e_noises, times, n_observations, actions=n_actions)
        q_samples = self.clip_return_support((e_noises + vels).squeeze(-1))
        qs = q_samples.mean(axis=1)

        return self.ensemble_agg(qs, self.config['q_agg'])

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss and metrics."""
        batch_size = batch['action_chunks'].shape[0]
        rng, noise_rng, gauss_rng, time_rng = jax.random.split(rng, 4)

        observations = batch['observations']
        actions = batch['action_chunks']
        next_observations = batch['chunk_next_observations']
        rewards = batch['chunk_rewards']
        masks = batch['chunk_masks']

        noises = jax.random.normal(noise_rng, (batch_size, 1))
        times = jax.random.uniform(time_rng, (batch_size, 1))
        num_vels = self.config['num_vels']

        # Scale rewards consistently with the return support.
        rewards = jnp.expand_dims(rewards, axis=-1)
        rewards = rewards * (1.0 - self.config['discount'] ** self.config['action_chunk_size'])
        discount = (self.config['discount'] ** self.config['action_chunk_size']) * jnp.expand_dims(masks, axis=-1)
        betas = (1.0 - times) * jnp.sqrt(jnp.maximum(1.0 - discount**2, 0.0))
        gauss = jax.random.normal(gauss_rng, (batch_size, 1))
        noise_scale = jnp.sqrt(jnp.maximum(1.0 - discount**2, 0.0))

        # Bootstrap from V at the next observation, without sampling actions.
        uncond_actions = self.network.select('cfg_uncond_embed')()[:, None]
        next_rets = self.compute_flow_returns(
            noises, next_observations, actions=uncond_actions, end_times=times, module_name='critic_flow'
        )
        noisy_next_rets = self.ensemble_agg(next_rets, self.config['ret_agg'])

        noisy_rets = times * rewards + discount * noisy_next_rets + betas * gauss

        vels = self.network.select('critic_flow')(
            jnp.broadcast_to(noisy_rets, (num_vels, *noisy_rets.shape)),
            times,
            observations,
            actions=actions,
            params=grad_params,
        )

        # Use the online value flow without gradient.
        target_next_vels = self.ensemble_agg(
            self.network.select('critic_flow')(
                jnp.broadcast_to(noisy_next_rets, (num_vels, *noisy_next_rets.shape)),
                times,
                next_observations,
                actions=uncond_actions,
            ),
            self.config['ret_agg'],
        )

        # Differentiate the interpolant, using v_V for the next-return derivative.
        target_vels = rewards - noise_scale * gauss + discount * target_next_vels

        per_sample_flow_loss = jnp.sum((vels - target_vels) ** 2, axis=0)
        chunk_valid = batch['chunk_valid']
        per_sample_flow_loss = per_sample_flow_loss * chunk_valid[:, None]
        flow_loss = jnp.mean(per_sample_flow_loss)

        critic_loss = flow_loss

        info = {
            'critic_loss': critic_loss,
            'flow_loss': flow_loss,
        }
        info['chunk_valid_frac'] = chunk_valid.mean()

        return critic_loss, info

    def value_loss(self, batch, grad_params, rng):
        """Compute the value loss and metrics."""
        batch_size = batch['action_chunks'].shape[0]
        rng, noise_rng, time_rng, adv_rng = jax.random.split(rng, 4)

        observations = batch['observations']
        actions = batch['action_chunks']

        noises = jax.random.normal(noise_rng, (batch_size, 1))
        times = jax.random.uniform(time_rng, (batch_size, 1))

        noisy_rets = self.ensemble_agg(
            self.compute_flow_returns(
                noises, observations, actions=actions, end_times=times, module_name='target_critic_flow'
            ),
            self.config['ret_agg'],
        )
        noisy_rets = jnp.broadcast_to(noisy_rets, (self.config['num_vels'], *noisy_rets.shape))

        target_vels = self.ensemble_agg(
            self.network.select('target_critic_flow')(noisy_rets, times, observations, actions=actions),
            self.config['ret_agg'],
        )

        uncond_actions = self.network.select('cfg_uncond_embed')(params=grad_params)[:, None]
        vels = self.network.select('critic_flow')(
            noisy_rets, times, observations, actions=uncond_actions, params=grad_params
        )

        # Q and V share noise to estimate the advantage.
        q = self.compute_qs_flow_t0(
            observations,
            adv_rng,
            actions=actions,
            module_name='target_critic_flow',
            num_ret_samples=self.config['adv_num_samples'],
        )
        v = self.compute_qs_flow_t0(observations, adv_rng, num_ret_samples=self.config['adv_num_samples'])
        adv = (q - v)[:, None]

        per_sample_value_loss = jnp.sum(self.expectile_loss(adv, target_vels - vels, self.config['expectile']), axis=0)
        chunk_valid = batch['chunk_valid']
        per_sample_value_loss = per_sample_value_loss * chunk_valid[:, None]
        value_loss = jnp.mean(per_sample_value_loss)

        frac_pos = (adv >= 0).mean()
        expectile_w_mean = self.config['expectile'] * frac_pos + (1.0 - self.config['expectile']) * (1.0 - frac_pos)
        adv_std = adv.std()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'q_mean': q.mean(),
            'adv_mean': adv.mean(),
            'adv_std': adv_std,
            'adv_abs_mean': jnp.abs(adv).mean(),
            'adv_snr': jnp.abs(adv.mean()) / (adv_std + 1e-8),
            'adv_frac_pos': frac_pos,
            'expectile_w_mean': expectile_w_mean,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss and metrics."""
        batch_size, ac_action_dim = batch['action_chunks'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        observations = batch['observations']
        actions = batch['action_chunks']

        noises = jax.random.normal(x_rng, (batch_size, ac_action_dim))
        times = jax.random.uniform(t_rng, (batch_size, 1))
        noisy_actions = (1 - times) * noises + times * actions
        velocity_fields = actions - noises

        velocity_field_preds = self.network.select('actor_flow')(
            observations, actions=noisy_actions, times=times, params=grad_params
        )
        per_sample_bc_loss = (velocity_field_preds - velocity_fields) ** 2
        chunk_valid = batch['chunk_valid']
        per_sample_bc_loss = per_sample_bc_loss * chunk_valid[:, None]
        flow_matching_loss = jnp.mean(per_sample_bc_loss)

        actor_loss = flow_matching_loss
        info = {
            'actor_loss': actor_loss,
            'bc_flow_loss': flow_matching_loss,
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
        tau = self.config['tau']
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        new_params = {**network.params, f'modules_target_{module_name}': new_target_params}
        return network.replace(params=new_params)

    @jax.jit
    def update(self, batch):
        """Return the updated agent and training metrics."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            """Evaluate the training loss for the candidate parameters."""
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn, log_grad_stats=self.config['log_grad_stats'])
        new_network = self.target_update(new_network, 'critic_flow')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('module_name',))
    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        end_times=None,
        module_name='critic_flow',
    ):
        """Euler-integrate ensemble return flows from shared noise of shape (*batch, 1).

        Returns have shape (num_vels, *batch, 1). Clip only at t=1 (`end_times=None`), never at intermediate times.
        """
        noisy_returns = jnp.broadcast_to(noises, (self.config['num_vels'], *noises.shape))
        clip_to_clean_return = end_times is None
        if end_times is None:
            end_times = jnp.ones((*noises.shape[:-1], 1), dtype=noises.dtype)
        step_size = end_times / self.config['critic_num_flow_steps']

        def func(carry, i):
            """Advance the flow by one Euler step."""
            (noisy_returns,) = carry
            times = i * step_size
            velocity_fields = self.network.select(module_name)(noisy_returns, times, observations, actions=actions)
            new_noisy_returns = noisy_returns + step_size * velocity_fields
            return (new_noisy_returns,), None

        (noisy_returns,), _ = jax.lax.scan(func, (noisy_returns,), jnp.arange(self.config['critic_num_flow_steps']))

        if clip_to_clean_return:
            noisy_returns = self.clip_return_support(noisy_returns)

        return noisy_returns

    @jax.jit
    def compute_flow_actions(
        self,
        noises,
        observations,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        noisy_actions = noises
        step_size = jnp.ones((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype) / self.config['num_flow_steps']

        def func(carry, i):
            """Advance the flow by one Euler step."""
            (noisy_actions,) = carry

            times = i * step_size
            velocity_fields = self.network.select('actor_flow')(observations, noisy_actions, times)
            new_noisy_actions = noisy_actions + velocity_fields * step_size

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
        action_seed, q_seed = jax.random.split(seed, 2)

        actor_noises = jax.random.normal(
            action_seed,
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
        flow_actions = self.compute_flow_actions(actor_noises, n_observations)

        # Share return noise across candidates to reduce ranking variance.
        leading_shape = observations.shape[: -len(self.config['ob_dims'])]
        n_noises = jax.random.normal(q_seed, (self.config['num_ret_samples'], *leading_shape, 1, 1))
        n_noises = jnp.broadcast_to(
            n_noises, (self.config['num_ret_samples'], *leading_shape, self.config['num_samples'], 1)
        )

        score = self.compute_qs(n_observations, actions=flow_actions, module_name='critic_flow', noises=n_noises)

        best_sample_idx = jnp.argmax(score, axis=-1)
        actions = jnp.take_along_axis(
            flow_actions,
            best_sample_idx[..., None, None],
            axis=-2,
        )
        actions = jnp.squeeze(actions, axis=-2)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Initialize the agent from an example batch and configuration."""
        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        mlp_class = MLP_CLASSES[config['mlp_class']]

        assert 'action_chunks' in example_batch, (
            'implicit_value_forcing needs chunked batches: set '
            'dataset.action_chunk_size (utils.dataset_subsampling.prepare_dataset does this from '
            "config['action_chunk_size']) before sampling the example batch."
        )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_action_chunks = example_batch['action_chunks']
        ex_times = ex_action_chunks[..., :1]
        ex_returns = jnp.broadcast_to(ex_action_chunks[..., :1], (config['num_vels'], *ex_action_chunks.shape[:-1], 1))
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        ac_action_dim = ex_action_chunks.shape[-1]
        assert ac_action_dim == config['action_chunk_size'] * action_dim, (
            f'action_chunks dim {ac_action_dim} != '
            f'action_chunk_size {config["action_chunk_size"]} * action_dim {action_dim}'
        )

        num_vels = config['num_vels']

        def make_return_flow():
            """Build a return-flow ensemble from the agent configuration."""
            return ValueVectorField(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['value_layer_norm'],
                num_ensembles=num_vels,
                time_embedding=config['time_embedding'],
                time_embed_dim=config['time_embed_dim'],  # Sinusoidal embedding dimension.
                mlp_class=mlp_class,
            )

        critic_flow_def = make_return_flow()
        target_critic_flow_def = make_return_flow()
        # One learned null action per member.
        cfg_uncond_embed_def = UnconditionalEmbedding(embed_dim=ac_action_dim, num_ensembles=num_vels)

        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,  # Set from the example batch.
            layer_norm=config['actor_layer_norm'],
            time_embedding=config['time_embedding'],
            time_embed_dim=config['time_embed_dim'],  # Sinusoidal embedding dimension.
            mlp_class=mlp_class,
        )

        network_info = dict(
            critic_flow=(critic_flow_def, (ex_returns, ex_times, ex_observations, ex_action_chunks)),
            target_critic_flow=(target_critic_flow_def, (ex_returns, ex_times, ex_observations, ex_action_chunks)),
            actor_flow=(actor_flow_def, (ex_observations, ex_action_chunks, ex_times)),
            cfg_uncond_embed=(cfg_uncond_embed_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        rng, init_rng = jax.random.split(rng)
        network_params = network_def.init(init_rng, **network_args)['params']

        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        target_params = {
            'modules_target_critic_flow': network.params['modules_critic_flow'],
        }
        network = network.replace(params={**network.params, **target_params})

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['ac_action_dim'] = ac_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the default agent configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='implicit_value_forcing',  # Agent registry key.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set from the example batch.
            action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            ac_action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            min_reward=ml_collections.config_dict.placeholder(float),  # Set from the dataset.
            max_reward=ml_collections.config_dict.placeholder(float),  # Set from the dataset.
            lr=3e-4,  # Optimizer learning rate.
            batch_size=1024,  # Training batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor hidden layer widths.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Shared critic and value flow.
            actor_layer_norm=True,  # Actor layer normalization.
            value_layer_norm=True,  # Shared critic and value flow layer normalization.
            mlp_class='mlp',  # 'mlp' or 'resmlp'.
            time_embedding='sinusoidal',  # 'raw' or 'sinusoidal'.
            time_embed_dim=16,  # Sinusoidal embedding dimension.
            discount=0.999,  # Per environment step.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile; 0.8 for cube-quadruple and humanoidmaze-giant in the paper.
            num_vels=2,  # Return flow ensemble size.
            q_agg='mean',  # Q/V ensemble aggregation: 'mean' or 'min'.
            ret_agg='mean',  # Noisy-return and target-velocity aggregation: 'mean' or 'min'.
            adv_num_samples=16,  # Return samples for the advantage estimate.
            clip_returns=True,  # Clip estimated returns to empirical return bounds.
            num_samples=16,  # Action candidates for best-of-N sampling.
            num_ret_samples=16,  # Return samples per Q estimate.
            num_flow_steps=10,  # Actor Euler steps.
            critic_num_flow_steps=10,  # Return flow Euler steps.
            action_chunk_size=5,  # Prediction horizon.
            action_chunk_eval_size=ml_collections.config_dict.placeholder(int),  # Actions to execute; None uses all.
            log_grad_stats=False,  # Log grad/max, grad/min, and grad/norm.
        )
    )

    return config
