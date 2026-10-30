import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP_CLASSES, ActorVectorField, Value


class QCAgent(flax.struct.PyTreeNode):
    """QC with a scalar critic and best-of-N action selection.

    https://arxiv.org/abs/2507.07969
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def aggregate_q(self, qs):
        """Aggregate critic members by minimum or mean."""
        # Value omits the ensemble axis when there is only one critic.
        if self.config['num_critics'] == 1:
            return qs
        if self.config['q_agg'] == 'min':
            return qs.min(axis=0)
        return qs.mean(axis=0)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss and metrics."""
        _, sample_rng = jax.random.split(rng)
        next_observations = batch['chunk_next_observations']
        # The online critic picks the next action; the target critic values it.
        next_actions = self.sample_actions(next_observations, seed=sample_rng)
        next_q = self.aggregate_q(self.network.select('target_critic')(next_observations, actions=next_actions))
        discount = self.config['discount'] ** self.config['action_chunk_size']
        target_q = jax.lax.stop_gradient(batch['chunk_rewards'] + discount * batch['chunk_masks'] * next_q)

        qs = self.network.select('critic')(batch['observations'], actions=batch['action_chunks'], params=grad_params)
        per_sample_loss = jnp.square(qs - target_q)
        if self.config['num_critics'] > 1:
            per_sample_loss = per_sample_loss.sum(axis=0)
        chunk_valid = batch['chunk_valid']
        per_sample_loss = per_sample_loss * chunk_valid
        critic_loss = per_sample_loss.mean()

        info = {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'target_q_mean': target_q.mean(),
        }
        info['chunk_valid_frac'] = chunk_valid.mean()
        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss and metrics."""
        batch_size, ac_action_dim = batch['action_chunks'].shape
        _, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, ac_action_dim))
        x_1 = batch['action_chunks']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        velocity = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        per_sample_loss = jnp.mean(jnp.square(pred - velocity), axis=-1)
        chunk_valid = batch['chunk_valid']
        per_sample_loss = per_sample_loss * chunk_valid
        actor_loss = per_sample_loss.mean()
        return actor_loss, {'actor_loss': actor_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the combined training loss and component metrics."""
        rng = rng if rng is not None else self.rng
        _, actor_rng, critic_rng = jax.random.split(rng, 3)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info = {f'critic/{k}': v for k, v in critic_info.items()}
        info.update({f'actor/{k}': v for k, v in actor_info.items()})
        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        """Soft-update the target network."""
        network.params[f'modules_target_{module_name}'] = jax.tree_util.tree_map(
            lambda p, tp: self.config['tau'] * p + (1 - self.config['tau']) * tp,
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )

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
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Sample actions from the policy."""
        _, action_seed = jax.random.split(seed)
        num_samples = self.config['num_samples']
        actions = jax.random.normal(action_seed, (*observations.shape[:-1], num_samples, self.config['ac_action_dim']))
        n_observations = jnp.repeat(observations[..., None, :], num_samples, axis=-2)

        def euler_step(i, actions):
            """Advance actions by one Euler step."""
            times = jnp.full((*actions.shape[:-1], 1), i / self.config['flow_steps'])
            velocities = self.network.select('actor_flow')(n_observations, actions, times)
            return actions + velocities / self.config['flow_steps']

        actions = jax.lax.fori_loop(0, self.config['flow_steps'], euler_step, actions)
        actions = jnp.clip(actions, -1, 1)

        q = self.aggregate_q(self.network.select('critic')(n_observations, actions=actions))
        indices = jnp.argmax(q, axis=-1)
        return jnp.take_along_axis(actions, indices[..., None, None], axis=-2).squeeze(axis=-2)

    @classmethod
    def create(cls, seed, example_batch, config):
        """Initialize the agent from an example batch and configuration."""
        if 'action_chunks' not in example_batch:
            raise ValueError('qc needs chunked batches: set dataset.action_chunk_size before sampling.')
        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        mlp_class = MLP_CLASSES[config['mlp_class']]

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_action_chunks = example_batch['action_chunks']
        action_dim = ex_actions.shape[-1]
        ac_action_dim = config['action_chunk_size'] * action_dim
        if ex_action_chunks.ndim != 2 or ex_action_chunks.shape != (ex_actions.shape[0], ac_action_dim):
            raise ValueError(
                f'action_chunks must have shape (batch_size, {ac_action_dim}), got {ex_action_chunks.shape}'
            )

        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_critics'],
            mlp_class=mlp_class,
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
            mlp_class=mlp_class,
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_action_chunks)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_action_chunks)),
            actor_flow=(actor_flow_def, (ex_observations, ex_action_chunks, ex_actions[..., :1])),
        )
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_args = {k: v[1] for k, v in network_info.items()}
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['ob_dims'] = ex_observations.shape[1:]
        config['action_dim'] = action_dim
        config['ac_action_dim'] = ac_action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the default agent configuration."""
    return ml_collections.ConfigDict(
        dict(
            agent_name='qc',  # Agent registry key.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set from the example batch.
            action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            ac_action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            lr=3e-4,  # Optimizer learning rate.
            batch_size=1024,  # Training batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor hidden layer widths.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Critic network.
            layer_norm=True,  # Critic layer normalization.
            actor_layer_norm=True,  # Actor layer normalization.
            mlp_class='mlp',  # 'mlp' or 'resmlp'.
            discount=0.999,  # Per environment step.
            tau=0.005,  # Target network update rate.
            num_critics=2,  # Critic ensemble size.
            q_agg='mean',  # Critic aggregation for ranking and targets: 'mean' or 'min'.
            num_samples=16,  # Best-of-N candidates for training and inference.
            flow_steps=10,  # Actor Euler steps.
            action_chunk_size=5,  # Prediction horizon.
            action_chunk_eval_size=ml_collections.config_dict.placeholder(int),  # Actions to execute; None uses all.
            log_grad_stats=False,  # Log grad/max, grad/min, and grad/norm.
        )
    )
