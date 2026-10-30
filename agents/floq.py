from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    ActorVectorField,
    FloQValueVectorField,
    Value,
)


class FloQAgent(flax.struct.PyTreeNode):
    """FloQ with a flow critic, a distilled scalar critic, and a one-step policy.

    https://arxiv.org/abs/2509.06863
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def repeat_tensor(x, n):
        """Repeat each batch entry `n` times contiguously."""
        batch_size, non_batch_dims = x.shape[0], x.shape[1:]
        return jnp.repeat(x, n, axis=0).reshape(batch_size * n, *non_batch_dims)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss and metrics."""
        batch_size = batch['action_chunks'].shape[0]
        repeat = self.config['noise_samples']
        chunk_valid = batch['chunk_valid']

        observations = self.repeat_tensor(batch['observations'], repeat)
        actions = self.repeat_tensor(batch['action_chunks'], repeat)
        next_observations = self.repeat_tensor(batch['chunk_next_observations'], repeat)

        rng, sample_rng = jax.random.split(rng, 2)
        next_actions = self.sample_actions(next_observations, seed=sample_rng)

        rng, next_noise_rng = jax.random.split(rng)
        next_noise_ratios = jax.random.uniform(next_noise_rng, (batch_size * repeat, 1))
        next_returns = (
            self.compute_target_flow_returns(
                next_observations,
                next_actions,
                noise_ratios=next_noise_ratios,
            )
            .squeeze(-1)
            .reshape((batch_size, repeat))
        )

        rewards = batch['chunk_rewards'].reshape((batch_size, 1))
        masks = batch['chunk_masks'].reshape((batch_size, 1))
        discount = self.config['discount'] ** self.config['action_chunk_size']
        target_returns = (rewards + discount * masks * next_returns).mean(axis=-1)

        rng, noise_rng = jax.random.split(rng)
        noise_ratios = jax.random.uniform(noise_rng, (batch_size * repeat, 1))
        current_returns = self.compute_current_flow_returns(
            observations,
            actions,
            noise_ratios=noise_ratios,
        )
        current_returns = current_returns.reshape((self.config['flow_num_ensembles'], batch_size, repeat))

        x_0 = (1.0 - noise_ratios) * self.config['noise_min'] + noise_ratios * self.config['noise_max']
        x_1 = jnp.tile(
            target_returns.reshape((1, batch_size, 1)),
            [self.config['flow_num_ensembles'], 1, repeat],
        )
        x_1 = x_1.reshape((self.config['flow_num_ensembles'], batch_size * repeat, 1))

        rng, t_rng = jax.random.split(rng, 2)
        t = jax.random.uniform(t_rng, (batch_size * repeat, 1))

        x_t = jnp.expand_dims(1.0 - t, axis=0) * jnp.expand_dims(x_0, axis=0) + jnp.expand_dims(t, axis=0) * x_1
        vel = x_1 - jnp.expand_dims(x_0, axis=0)

        pred = self.network.select('floq')(observations, actions, returns=x_t, times=t, params=grad_params)
        vel = vel.reshape((self.config['flow_num_ensembles'], batch_size, repeat))
        pred = pred.reshape((self.config['flow_num_ensembles'], batch_size, repeat))
        floq_loss = jnp.sum((pred - vel) ** 2, axis=-1)

        q = self.network.select('critic')(batch['observations'], actions=batch['action_chunks'], params=grad_params)
        mean_current_returns = current_returns.mean(axis=-1).mean(axis=0)
        distilled_critic_loss = jnp.square(q - mean_current_returns)

        floq_loss = floq_loss * chunk_valid
        distilled_critic_loss = distilled_critic_loss * chunk_valid

        # Flow and scalar critics may have different ensemble sizes.
        critic_loss = floq_loss.mean() + distilled_critic_loss.mean()

        cr = jnp.mean(current_returns, axis=0)
        var_flow_returns = jnp.mean(jnp.var(cr, axis=-1))
        td_error = jnp.mean((mean_current_returns - target_returns) ** 2)

        info = {
            'floq_loss': floq_loss.mean(),
            'distilled_critic_loss': distilled_critic_loss.mean(),
            'critic_loss': critic_loss.mean(),
            'q': q.mean(),
            'var_flow_returns': var_flow_returns,
            'td_error': td_error,
        }
        info['chunk_valid_frac'] = chunk_valid.mean()

        return critic_loss.mean(), info

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

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = (pred - vel) ** 2
        bc_flow_loss = bc_flow_loss * chunk_valid[:, None]

        # Policy distillation uses its own actions, so no dataset validity mask applies.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, ac_action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()

        actor_loss = bc_flow_loss.mean() + self.config['alpha'] * distill_loss + q_loss

        mse = jnp.mean((self.sample_actions(batch['observations'], seed=rng) - batch['action_chunks']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss.mean(),
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the combined training loss and component metrics."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

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
        return network.replace(
            params={
                **network.params,
                f'modules_target_{module_name}': new_target_params,
            }
        )

    @jax.jit
    def update(self, batch):
        """Return the updated agent and training metrics."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            """Evaluate the training loss for the candidate parameters."""
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn, log_grad_stats=self.config['log_grad_stats'])
        new_network = self.target_update(new_network, 'floq')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the policy."""
        del temperature
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['ac_action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def compute_target_flow_returns(
        self,
        observations,
        actions,
        noise_ratios,
    ):
        """Integrate the target critic flow from return-noise ratios and aggregate ensemble members."""
        assert observations.ndim == 2 and actions.ndim == 2
        assert noise_ratios.ndim == 2 and noise_ratios.shape[-1] == 1

        returns = self.config['noise_min'] * (1 - noise_ratios) + self.config['noise_max'] * noise_ratios
        returns = jnp.expand_dims(returns, axis=0)
        returns = jnp.tile(returns, [self.config['flow_num_ensembles'], 1, 1])

        for i in range(self.config['critic_flow_steps']):
            t = jnp.full(
                (*observations.shape[:-1], 1),
                i / self.config['critic_flow_steps'],
            )
            vels = self.network.select('target_floq')(
                observations,
                actions,
                returns=returns,
                times=t,
            )
            returns = returns + vels / self.config['critic_flow_steps']

        if self.config['q_agg'] == 'min':
            returns = jnp.min(returns, axis=0)
        else:
            returns = jnp.mean(returns, axis=0)

        return returns

    @jax.jit
    def compute_current_flow_returns(
        self,
        observations,
        actions,
        noise_ratios,
    ):
        """Integrate the online critic flow from return-noise ratios for each ensemble member."""
        assert observations.ndim == 2 and actions.ndim == 2
        assert noise_ratios.ndim == 2 and noise_ratios.shape[-1] == 1

        returns = self.config['noise_min'] * (1 - noise_ratios) + self.config['noise_max'] * noise_ratios
        returns = jnp.expand_dims(returns, axis=0)
        returns = jnp.tile(returns, [self.config['flow_num_ensembles'], 1, 1])

        for i in range(self.config['critic_flow_steps']):
            t = jnp.full(
                (*observations.shape[:-1], 1),
                i / self.config['critic_flow_steps'],
            )
            vels = self.network.select('floq')(
                observations,
                actions,
                returns=returns,
                times=t,
            )
            returns = returns + vels / self.config['critic_flow_steps']

        return returns

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        actions = noises
        for i in range(self.config['actor_flow_steps']):
            t = jnp.full(
                (*observations.shape[:-1], 1),
                i / self.config['actor_flow_steps'],
            )
            vels = self.network.select('actor_bc_flow')(observations, actions, t)
            actions = actions + vels / self.config['actor_flow_steps']
        return jnp.clip(actions, -1, 1)

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
            'floq needs chunked batches: set dataset.action_chunk_size (utils.dataset_subsampling.prepare_dataset '
            "does this from config['action_chunk_size']) before sampling the example batch."
        )

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_action_chunks = example_batch['action_chunks']
        critic_ex_times = ex_action_chunks[..., :1]
        actor_ex_times = ex_action_chunks[..., :1]
        ex_returns = ex_action_chunks[..., :1]
        ex_returns = jnp.tile(
            jnp.expand_dims(ex_returns, axis=0),
            [config['flow_num_ensembles'], 1, 1],
        )
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        ac_action_dim = ex_action_chunks.shape[-1]
        assert ac_action_dim == config['action_chunk_size'] * action_dim, (
            f'action_chunks dim {ac_action_dim} != '
            f'action_chunk_size {config["action_chunk_size"]} * action_dim {action_dim}'
        )

        r_min = float(config['min_reward'])
        r_max = float(config['max_reward'])

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['distill_num_ensembles'],
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
            time_embedding='raw',
        )

        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
            time_embedding='raw',
        )

        # Use the per-step discount; accumulated rewards are already discounted.
        q_min = (r_min - config['reward_offset']) / (1.0 - config['discount'])
        q_max = (r_max + config['reward_offset']) / (1.0 - config['discount'])

        floq_hidden_dims = config['block_depth'] * [config['block_width']]
        floq_def = FloQValueVectorField(
            hidden_dims=floq_hidden_dims,
            layer_norm=config['layer_norm'],
            num_ensembles=config['flow_num_ensembles'],
            embed_time=config['embed_time'],
            clip_return_embed=config['clip_return_embed'],
            q_min=q_min,
            q_max=q_max,
            num_bins=config['num_bins'],
            sigma=config['sigma'],
        )
        target_floq_def = FloQValueVectorField(
            hidden_dims=floq_hidden_dims,
            layer_norm=config['layer_norm'],
            num_ensembles=config['flow_num_ensembles'],
            embed_time=config['embed_time'],
            clip_return_embed=config['clip_return_embed'],
            q_min=q_min,
            q_max=q_max,
            num_bins=config['num_bins'],
            sigma=config['sigma'],
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_action_chunks)),
            floq=(floq_def, (ex_observations, ex_action_chunks, ex_returns, critic_ex_times)),
            target_floq=(
                target_floq_def,
                (ex_observations, ex_action_chunks, ex_returns, critic_ex_times),
            ),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_action_chunks, actor_ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_action_chunks, None)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        network = network.replace(
            params={
                **network.params,
                'modules_target_floq': network.params['modules_floq'],
            }
        )

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['ac_action_dim'] = ac_action_dim
        config['noise_min'] = config['noise_coverage'] * r_min / (1.0 - config['discount'])
        config['noise_max'] = config['noise_coverage'] * r_max / (1.0 - config['discount'])

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the default agent configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='floq',  # Agent registry key.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set automatically.
            action_dim=ml_collections.config_dict.placeholder(int),  # Set automatically.
            ac_action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            min_reward=ml_collections.config_dict.placeholder(float),  # Set from the dataset.
            max_reward=ml_collections.config_dict.placeholder(float),  # Set from the dataset.
            noise_min=ml_collections.config_dict.placeholder(float),  # Initial return lower bound; set automatically.
            noise_max=ml_collections.config_dict.placeholder(float),  # Initial return upper bound; set automatically.
            lr=3e-4,  # Optimizer learning rate.
            batch_size=1024,  # Training batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # BC flow and one-step policy.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Scalar critic.
            layer_norm=True,  # Flow and scalar critics.
            actor_layer_norm=False,  # BC flow and one-step policy.
            discount=0.999,  # Per environment step.
            tau=0.005,  # Target update rate.
            q_agg='min',  # Target flow ensemble aggregation: min or mean.
            alpha=300.0,  # Policy distillation weight; paper: 3000 on puzzle-4x4.
            actor_flow_steps=10,  # BC flow integration steps.
            flow_num_ensembles=2,  # Critic flow ensemble size.
            distill_num_ensembles=2,  # Scalar critic ensemble size.
            noise_samples=8,  # Initial return samples per batch entry.
            noise_coverage=0.1,  # Scale of reward bounds / (1 - discount); paper: 0.25 on puzzle-4x4.
            critic_flow_steps=8,  # Critic flow integration steps.
            embed_time=True,  # Use cos(t) instead of t.
            clip_return_embed=False,  # Clip to HL-Gauss support; paper: enabled only on cube-quadruple.
            num_bins=51,  # HL-Gauss support points.
            sigma=16.0,  # HL-Gauss width in bins.
            reward_offset=0.01,  # Reward margin for the HL-Gauss support.
            block_width=512,  # Critic flow hidden width.
            block_depth=2,  # Critic flow hidden layers; paper: 4 on puzzle-4x4 and humanoidmaze-giant.
            log_grad_stats=False,  # Log gradient max, min, and norm.
            action_chunk_size=5,  # Prediction horizon.
            action_chunk_eval_size=ml_collections.config_dict.placeholder(int),  # Actions to execute; None uses all.
        )
    )
    return config
