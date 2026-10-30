import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP_CLASSES, ActorVectorField, Value, categorical_support


class ImplicitC51Agent(flax.struct.PyTreeNode):
    """C51 with an expectile-weighted value distribution and best-of-N flow sampling.

    https://arxiv.org/abs/1707.06887

    Critic backups use the next-state value.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def value_loss(self, batch, grad_params):
        """Compute the value loss and metrics."""
        target_logits = self.network.select('target_critic')(batch['observations'], batch['action_chunks'])
        target_probs = jax.nn.softmax(target_logits, axis=-1)
        target_qs = jnp.sum(target_probs * self.config['atoms'], axis=-1)
        # For 'min', keep the distribution from the member with the lower expected return.
        if self.config['q_agg'] == 'min':
            member_idxs = jnp.argmin(target_qs, axis=0)
            batch_idxs = jnp.arange(target_qs.shape[1])
            target_probs = target_probs[member_idxs, batch_idxs]
            q = target_qs[member_idxs, batch_idxs]
        else:
            target_probs = target_probs.mean(axis=0)
            q = target_qs.mean(axis=0)

        value_logits = self.network.select('value')(batch['observations'], params=grad_params)
        value_probs = jax.nn.softmax(value_logits, axis=-1)
        v = jnp.sum(value_probs * self.config['atoms'], axis=-1)

        adv = q - v
        weights = jnp.where(adv >= 0, self.config['expectile'], 1 - self.config['expectile'])
        value_ce = optax.softmax_cross_entropy(value_logits, target_probs)
        per_sample_value_loss = weights * value_ce
        chunk_valid = batch['chunk_valid']
        per_sample_value_loss = per_sample_value_loss * chunk_valid
        value_loss = jnp.mean(per_sample_value_loss)

        frac_pos = (adv >= 0).mean()
        info = {
            'value_loss': value_loss,
            'value_ce': value_ce.mean(),
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'adv_mean': adv.mean(),
            'adv_frac_pos': frac_pos,
            'expectile_w_mean': weights.mean(),
            'entropy': -jnp.sum(value_probs * jnp.log(value_probs + 1e-8), axis=-1).mean(),
        }
        info['chunk_valid_frac'] = chunk_valid.mean()
        return value_loss, info

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss and metrics."""
        del rng

        logits = self.network.select('critic')(batch['observations'], batch['action_chunks'], params=grad_params)

        next_value_logits = self.network.select('value')(batch['chunk_next_observations'])
        next_probs = jax.nn.softmax(next_value_logits, axis=-1)

        rewards = batch['chunk_rewards']
        masks = batch['chunk_masks']
        discount = self.config['discount'] ** self.config['action_chunk_size']
        batch_size = rewards.shape[0]
        offset = jnp.arange(batch_size, dtype=jnp.int32) * self.config['num_atoms']
        offset = jnp.broadcast_to(offset[:, None], (batch_size, self.config['num_atoms']))

        # Project each target atom onto its two neighboring support atoms.
        projected_atoms = jnp.clip(
            rewards[:, None] + discount * masks[:, None] * self.config['atoms'][None],
            self.config['v_min'],
            self.config['v_max'],
        )
        projected_bins = (projected_atoms - self.config['v_min']) / self.config['delta_atom']
        lower_bins = jnp.floor(projected_bins).astype(jnp.int32)
        upper_bins = jnp.ceil(projected_bins).astype(jnp.int32)

        delta_mass_lower = (upper_bins + (lower_bins == upper_bins).astype(jnp.int32) - projected_bins) * next_probs
        delta_mass_upper = (projected_bins - lower_bins) * next_probs

        target_mass = jnp.zeros(batch_size * self.config['num_atoms'])
        target_mass = target_mass.at[(lower_bins + offset).ravel()].add(delta_mass_lower.ravel())
        target_mass = target_mass.at[(upper_bins + offset).ravel()].add(delta_mass_upper.ravel())
        target_mass = target_mass.reshape(batch_size, self.config['num_atoms'])

        target_mass = jnp.broadcast_to(target_mass, logits.shape)
        per_sample_critic_loss = optax.softmax_cross_entropy(logits, target_mass)
        chunk_valid = batch['chunk_valid']
        per_sample_critic_loss = per_sample_critic_loss * chunk_valid
        critic_loss = per_sample_critic_loss.mean()

        probs = jax.nn.softmax(logits, axis=-1)
        q = jnp.sum(probs * self.config['atoms'], axis=-1)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'entropy': -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1).mean(),
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
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params)
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
        n_noises = jax.random.normal(
            seed,
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

        n_logits = self.network.select('critic')(n_observations, n_actions)
        n_probs = jax.nn.softmax(n_logits, axis=-1)
        qs = jnp.sum(n_probs * self.config['atoms'][None, None], axis=-1)
        if self.config['q_agg'] == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

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
        v_min, v_max, atoms, delta_atom = categorical_support(
            config['min_reward'], config['max_reward'], config['discount'], config['num_atoms']
        )
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        assert 'action_chunks' in example_batch, (
            'implicit_c51 needs chunked batches: set dataset.action_chunk_size '
            "(utils.dataset_subsampling.prepare_dataset does this from config['action_chunk_size']) before sampling "
            'the example batch.'
        )
        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        mlp_class = MLP_CLASSES[config['mlp_class']]

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
            mlp_class=mlp_class,
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            value_dim=config['num_atoms'],
            layer_norm=config['value_layer_norm'],
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

        # Atom support uses the per-step discount.
        config['v_min'] = v_min
        config['v_max'] = v_max
        config['atoms'] = atoms
        config['delta_atom'] = delta_atom

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['ac_action_dim'] = ac_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the default agent configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='implicit_c51',  # Agent registry key.
            min_reward=ml_collections.config_dict.placeholder(float),  # Empirical minimum; set automatically.
            max_reward=ml_collections.config_dict.placeholder(float),  # Empirical maximum; set automatically.
            v_min=ml_collections.config_dict.placeholder(float),  # Lowest atom; set automatically.
            v_max=ml_collections.config_dict.placeholder(float),  # Highest atom; set automatically.
            atoms=ml_collections.config_dict.placeholder(jnp.ndarray),  # Return support; set automatically.
            delta_atom=ml_collections.config_dict.placeholder(float),  # Atom spacing; set automatically.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set from the example batch.
            action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            ac_action_dim=ml_collections.config_dict.placeholder(int),  # Set from the example batch.
            lr=3e-4,  # Optimizer learning rate.
            batch_size=1024,  # Training batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor hidden layer widths.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value and critic hidden layer widths.
            value_layer_norm=True,  # Value and critic layer normalization.
            actor_layer_norm=True,  # Actor layer normalization.
            mlp_class='mlp',  # 'mlp' or 'resmlp'.
            num_atoms=51,  # Return distribution atoms.
            discount=0.999,  # Per environment step.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile; 0.8 for cube-quadruple and humanoidmaze-giant in the paper.
            q_agg='min',  # Critic ensemble aggregation: 'min' or 'mean'.
            num_samples=16,  # Action candidates for best-of-N sampling.
            num_flow_steps=10,  # Actor Euler steps.
            log_grad_stats=False,  # Log grad/max, grad/min, and grad/norm.
            action_chunk_size=5,  # Prediction horizon.
            action_chunk_eval_size=ml_collections.config_dict.placeholder(int),  # Actions to execute; None uses all.
        )
    )
    return config
