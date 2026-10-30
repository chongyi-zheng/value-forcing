from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP_CLASSES, ValueVectorField, ActorVectorField


class ValueFlowsAgent(flax.struct.PyTreeNode):
    """Value Flows with best-of-N selection using one-step Q estimates.

    https://arxiv.org/abs/2510.07650

    The critic combines confidence-weighted bootstrapped and distributional flow matching.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def aggregate_q(self, qs, config_key='q_agg'):
        """Aggregate critic members by minimum or mean."""
        if self.config[config_key] == 'min':
            return qs.min(axis=0)
        return qs.mean(axis=0)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss and metrics."""
        batch_size = batch['action_chunks'].shape[0]
        rng, actor_rng, noise_rng, time_rng, q_rng, ret_rng = jax.random.split(rng, 6)

        observations = batch['observations']
        action_chunks = batch['action_chunks']
        next_observations = batch['chunk_next_observations']
        chunk_valid = batch['chunk_valid']
        num_critics = self.config['num_critics']
        discount = self.config['discount'] ** self.config['action_chunk_size']

        next_actions = self.sample_actions(next_observations, actor_rng)

        # Confidence uses sigma = |dZ/dz^0| of the target flow at dataset actions.
        ret_noises = jax.random.normal(ret_rng, (batch_size, 1))
        _, ret_jac_eps_prods = self.compute_flow_returns(
            jnp.broadcast_to(ret_noises, (num_critics, *ret_noises.shape)),
            observations,
            action_chunks,
            flow_network_name='target_critic_flow',
            return_jac_eps_prod=True,
        )
        ret_stds = self.aggregate_q(jnp.abs(ret_jac_eps_prods.squeeze(-1)))
        weights = jax.nn.sigmoid(-self.config['confidence_weight_temp'] / ret_stds) + 0.5
        weights = jax.lax.stop_gradient(weights)

        noises = jax.random.normal(noise_rng, (batch_size, 1))
        times = jax.random.uniform(time_rng, (batch_size, 1))
        # Share noise and batch the BCFM (t=1) and DCFM (t=times) targets in one scan.
        two_noises = jnp.concatenate([noises, noises], axis=0)
        two_next_obs = jnp.concatenate([next_observations, next_observations], axis=0)
        two_next_actions = jnp.concatenate([next_actions, next_actions], axis=0)
        two_end_times = jnp.concatenate([jnp.ones_like(times), times], axis=0)
        merged_returns = self.compute_flow_returns(
            jnp.broadcast_to(two_noises, (num_critics, *two_noises.shape)),
            two_next_obs,
            two_next_actions,
            end_times=two_end_times,
            flow_network_name='target_critic_flow',
        )
        next_returns_e = merged_returns[:, :batch_size]
        noisy_next_returns_e = merged_returns[:, batch_size:]
        next_returns = self.aggregate_q(next_returns_e, config_key='ret_agg')

        returns = (
            jnp.expand_dims(batch['chunk_rewards'], axis=-1)
            + discount * jnp.expand_dims(batch['chunk_masks'], axis=-1) * next_returns
        )
        noisy_returns = times * returns + (1 - times) * noises
        target_vector_field = returns - noises

        # Sum squared errors over critic members.
        vector_fields = self.network.select('critic_flow')(
            jnp.broadcast_to(noisy_returns, (num_critics, *noisy_returns.shape)),
            times,
            observations,
            action_chunks,
            params=grad_params,
        )
        bcfm_loss = ((vector_fields - target_vector_field) ** 2).sum(axis=0).mean(axis=-1)

        noisy_next_returns = self.aggregate_q(noisy_next_returns_e, config_key='ret_agg')
        noisy_returns = (
            jnp.expand_dims(batch['chunk_rewards'], axis=-1)
            + discount * jnp.expand_dims(batch['chunk_masks'], axis=-1) * noisy_next_returns
        )
        vector_fields = self.network.select('critic_flow')(
            jnp.broadcast_to(noisy_returns, (num_critics, *noisy_returns.shape)),
            times,
            observations,
            action_chunks,
            params=grad_params,
        )
        target_vector_fields = self.network.select('target_critic_flow')(
            jnp.broadcast_to(noisy_next_returns, (num_critics, *noisy_next_returns.shape)),
            times,
            next_observations,
            next_actions,
        )
        target_vector_field = self.aggregate_q(target_vector_fields, config_key='ret_agg')
        dcfm_loss = ((vector_fields - target_vector_field) ** 2).sum(axis=0).mean(axis=-1)

        critic_loss = self.config['bcfm_lambda'] * bcfm_loss + self.config['dcfm_lambda'] * dcfm_loss
        critic_loss = critic_loss * chunk_valid
        critic_loss = (weights * critic_loss).mean()

        q_noises = jax.random.normal(q_rng, (batch_size, 1))
        qs_noises = jnp.broadcast_to(q_noises, (num_critics, *q_noises.shape))
        qs = (
            qs_noises
            + self.network.select('critic_flow')(qs_noises, jnp.zeros_like(qs_noises), observations, action_chunks)
        ).squeeze(-1)
        if self.config['clip_flow_returns']:
            qs = jnp.clip(
                qs,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
        q = self.aggregate_q(qs)
        q_stds = ret_stds

        info = {
            'critic_loss': critic_loss,
            'bcfm_loss': bcfm_loss.mean(),
            'dcfm_loss': dcfm_loss.mean(),
            'q_mean': q.mean(),
            'q_std': q_stds.mean(),
            'q_std_max': q_stds.max(),
            'q_std_min': q_stds.min(),
            'q_max': q.max(),
            'q_min': q.min(),
            'weight': weights.mean(),
        }
        info['chunk_valid_frac'] = chunk_valid.mean()

        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss and metrics."""
        batch_size, ac_action_dim = batch['action_chunks'].shape
        chunk_valid = batch['chunk_valid']
        # Preserve the five-way split to keep RNG streams unchanged under legacy JAX settings.
        rng, x_rng, t_rng, _, _ = jax.random.split(rng, 5)

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

    @partial(jax.jit, static_argnames=('flow_network_name', 'return_jac_eps_prod'))
    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        end_times=None,
        flow_network_name='critic_flow',
        return_jac_eps_prod=False,
    ):
        """Integrate the return flow to `end_times` (default 1), with optional return clipping.

        Optionally return dZ/dz^0 propagated through Euler steps without differentiating the clip.
        """
        noisy_returns = noises
        noisy_jac_eps_prod = jnp.ones_like(noises)
        init_times = jnp.zeros((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            """Advance the flow by one Euler step."""
            (noisy_returns, noisy_jac_eps_prod) = carry

            times = i * step_size + init_times
            vector_field, jac_eps_prod = jax.jvp(
                lambda ret: self.network.select(flow_network_name)(ret, times, observations, actions),
                (noisy_returns,),
                (noisy_jac_eps_prod,),
            )

            new_noisy_returns = noisy_returns + step_size * vector_field
            new_noisy_jac_eps_prod = noisy_jac_eps_prod + step_size * jac_eps_prod
            if self.config['clip_flow_returns']:
                new_noisy_returns = jnp.clip(
                    new_noisy_returns,
                    self.config['min_reward'] / (1 - self.config['discount']),
                    self.config['max_reward'] / (1 - self.config['discount']),
                )

            return (new_noisy_returns, new_noisy_jac_eps_prod), None

        (noisy_returns, noisy_jac_eps_prod), _ = jax.lax.scan(
            func, (noisy_returns, noisy_jac_eps_prod), jnp.arange(self.config['num_flow_steps'])
        )

        if return_jac_eps_prod:
            return noisy_returns, noisy_jac_eps_prod
        else:
            return noisy_returns

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
            new_noisy_actions = jnp.clip(new_noisy_actions, -1, 1)

            return (new_noisy_actions,), None

        (noisy_actions,), _ = jax.lax.scan(func, (noisy_actions,), jnp.arange(self.config['num_flow_steps']))

        return noisy_actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the policy."""
        action_seed, q_seed = jax.random.split(seed)

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

        q_noises = jax.random.normal(
            q_seed, (*observations.shape[: -len(self.config['ob_dims'])], self.config['num_samples'], 1)
        )

        qs_noises = jnp.broadcast_to(q_noises, (self.config['num_critics'], *q_noises.shape))
        qs = (
            qs_noises
            + self.network.select('critic_flow')(qs_noises, jnp.zeros_like(qs_noises), n_observations, flow_actions)
        ).squeeze(-1)
        if self.config['clip_flow_returns']:
            qs = jnp.clip(
                qs,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
        q = self.aggregate_q(qs)

        if len(q.shape) > 1:
            actions = flow_actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            actions = flow_actions[jnp.argmax(q, axis=-1)]

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Initialize the agent from an example batch and configuration."""
        assert 'action_chunks' in example_batch, (
            'value_flows needs chunked batches: set dataset.action_chunk_size '
            "(utils.dataset_subsampling.prepare_dataset does this from config['action_chunk_size']) before sampling "
            'the example batch.'
        )
        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        mlp_class = MLP_CLASSES[config['mlp_class']]

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_action_chunks = example_batch['action_chunks']
        num_critics = config['num_critics']
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
            mlp_class=mlp_class,
        )
        target_critic_flow_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=num_critics,
            mlp_class=mlp_class,
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
            mlp_class=mlp_class,
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
            agent_name='value_flows',  # Agent registry key.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set automatically.
            action_dim=ml_collections.config_dict.placeholder(int),  # Set automatically.
            ac_action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            min_reward=ml_collections.config_dict.placeholder(float),  # Set from the dataset.
            max_reward=ml_collections.config_dict.placeholder(float),  # Set from the dataset.
            lr=3e-4,  # Optimizer learning rate.
            batch_size=1024,  # Training batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor hidden layer widths.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Critic flow hidden layer widths.
            actor_layer_norm=True,  # Actor layer normalization.
            value_layer_norm=True,  # Critic flow layer normalization.
            mlp_class='mlp',  # mlp or resmlp.
            discount=0.999,  # Per environment step.
            tau=0.005,  # Target update rate.
            num_critics=2,  # Critic flow ensemble size.
            ret_agg='min',  # Target return and velocity ensemble aggregation: min or mean.
            q_agg='min',  # Q and confidence ensemble aggregation: min or mean.
            clip_flow_returns=True,  # Clip to [min_reward, max_reward] / (1 - discount).
            confidence_weight_temp=0.3,  # Confidence temperature; paper: 1.0 on puzzle-4x4.
            dcfm_lambda=1.0,  # Distributional flow matching weight.
            bcfm_lambda=0.3,  # Bootstrapped flow matching weight; paper: 3.0 on puzzle-4x4 and humanoidmaze-giant.
            num_samples=16,  # Candidates for best-of-N action selection.
            num_flow_steps=10,  # Actor and critic flow integration steps.
            log_grad_stats=False,  # Log gradient max, min, and norm.
            action_chunk_size=5,  # Prediction horizon.
            action_chunk_eval_size=ml_collections.config_dict.placeholder(int),  # Actions to execute; None uses all.
        )
    )
    return config
