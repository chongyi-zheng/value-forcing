import math
from typing import Any, Sequence

import jax
import flax.linen as nn
import jax.numpy as jnp


jax_float_dtype = jax.dtypes.canonicalize_dtype(jnp.float64)


def default_init(scale=1.0):
    """Return a uniform variance-scaling kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def categorical_support(min_reward, max_reward, discount, num_atoms):
    """Build the categorical return support, rejecting bounds that make projection undefined."""
    if num_atoms < 2:
        raise ValueError(f'Categorical critics require at least 2 atoms; got {num_atoms}.')
    if not 0 <= discount < 1:
        raise ValueError(f'Categorical critics require discount in [0, 1); got {discount}.')
    if not (math.isfinite(min_reward) and math.isfinite(max_reward) and min_reward < max_reward):
        raise ValueError(
            'Categorical critics require finite reward bounds with min_reward < max_reward; '
            f'got min_reward={min_reward}, max_reward={max_reward}. '
            'Constant-reward datasets give a zero-width support and cannot be projected.'
        )
    v_min = min_reward / (1 - discount)
    v_max = max_reward / (1 - discount)
    atoms = jnp.linspace(v_min, v_max, num_atoms)
    delta_atom = (v_max - v_min) / (num_atoms - 1)
    return v_min, v_max, atoms, delta_atom


def timestep_embedding(t, emb_size=16, max_period=10000):
    """Embed times in [0, 1] with the sinusoidal features from kvfrans/cfgrl `DiffusionPolicy`.

    Map (..., 1) to (..., emb_size), where emb_size is even; scale times by max_period.
    """
    t = t.astype(jax_float_dtype) * max_period
    half = emb_size // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jax_float_dtype) / half)
    args = t * freqs
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


def embed_time(t, time_embedding: str, emb_size=16):
    """Return sinusoidal or raw time features, expanding (batch,) inputs to (batch, 1)."""
    t = jax.lax.convert_element_type(t, jax_float_dtype)
    if t.ndim == 1:
        t = t[..., None]
    if time_embedding == 'sinusoidal':
        return timestep_embedding(t, emb_size=emb_size)
    if time_embedding == 'raw':
        return t
    raise ValueError(f"Unknown time_embedding {time_embedding!r}; expected 'sinusoidal' or 'raw'.")


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Vectorize a module with independent parameters for each ensemble member."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multilayer perceptron with optional layer normalization and final activation."""

    hidden_dims: Sequence[int]
    activation: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        """Apply dense layers with the configured activations and normalization."""
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) and self.layer_norm:
                x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activation(x)
        return x


class ResidualMLP(nn.Module):
    """Residual MLP from the OCBench reference implementations.

    Use an even number of equal-width hidden layers; specify the output width separately.
    """

    hidden_dims: Sequence[int]
    output_dim: int
    activation: Any = nn.gelu
    kernel_init: Any = default_init()
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        """Apply the input layer, residual blocks, and linear output layer."""
        if len(self.hidden_dims) % 2:
            raise ValueError(f'ResidualMLP requires an even number of hidden layers; got {len(self.hidden_dims)}.')
        if not self.hidden_dims:
            return nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)

        hidden_dim = self.hidden_dims[0]
        if any(dim != hidden_dim for dim in self.hidden_dims):
            raise ValueError(f'ResidualMLP requires constant hidden widths; got {self.hidden_dims}.')

        x = nn.Dense(hidden_dim, kernel_init=self.kernel_init)(x)
        x = self.activation(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)

        for _ in range(len(self.hidden_dims) // 2):
            residual = x
            x = nn.Dense(hidden_dim, kernel_init=self.kernel_init)(x)
            x = self.activation(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.Dense(hidden_dim, kernel_init=self.kernel_init)(x)
            x = residual + x
            x = self.activation(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)

        return nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)


MLP_CLASSES = dict(
    mlp=MLP,
    resmlp=ResidualMLP,
)


class CosineEmbedding(nn.Module):
    """Cosine features for quantile fractions."""

    num_cosines: int = 64

    @nn.compact
    def __call__(self, taus):
        """Embed quantile fractions with integer cosine frequencies."""
        freqs = jnp.pi * jnp.arange(1, self.num_cosines + 1)[None, None]
        cos_embeddings = jnp.cos(freqs * taus[..., None])

        return cos_embeddings


class Value(nn.Module):
    """State-value V(s) or action-value Q(s, a) network with optional ensembling."""

    hidden_dims: Sequence[int]
    value_dim: int = 1
    layer_norm: bool = True
    num_ensembles: int = 2
    mlp_class: Any = MLP

    def setup(self):
        """Initialize the value network and optional ensemble."""
        mlp_class = self.mlp_class
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        if self.mlp_class is ResidualMLP:
            value_net = mlp_class(self.hidden_dims, self.value_dim, layer_norm=self.layer_norm)
        else:
            value_net = mlp_class((*self.hidden_dims, self.value_dim), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Predict state values, or action values when actions are supplied."""
        inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        if self.value_dim == 1:
            v = v.squeeze(-1)

        return v


class ValueVectorField(nn.Module):
    """Return vector field with raw or sinusoidal time features.

    For multiple ensemble members, returns carry a leading ensemble axis and each member reads its own slice.
    Other inputs broadcast to the return tensor's ensemble and batch dimensions.
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    layer_norm: bool = False
    num_ensembles: int = 2
    time_embedding: str = 'sinusoidal'
    time_embed_dim: int = 16
    mlp_class: Any = MLP

    def setup(self) -> None:
        """Initialize the return network with separate inputs for each ensemble member."""
        mlp_class = self.mlp_class
        if self.num_ensembles > 1:
            # Each member consumes its own return and optional null-action slice.
            mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=0)
        if self.mlp_class is ResidualMLP:
            value_net = mlp_class(self.hidden_dims, self.value_dim, layer_norm=self.layer_norm)
        else:
            value_net = mlp_class((*self.hidden_dims, self.value_dim), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    @nn.compact
    def __call__(self, returns, times, observations, actions=None):
        """Predict return velocities after broadcasting inputs to the return tensor's leading dimensions.

        For num_ensembles > 1, returns must have shape (num_ensembles, *batch, 1).
        """
        if self.num_ensembles > 1 and returns.shape[0] != self.num_ensembles:
            raise ValueError(
                f'returns must carry a leading ensemble axis of size '
                f'{self.num_ensembles} when num_ensembles > 1; got shape '
                f'{returns.shape}. Broadcast a shared return tensor with '
                f'jnp.broadcast_to(returns, (num_ensembles, *returns.shape)).'
            )
        lead_shape = returns.shape[:-1]

        def broadcast_lead(x):
            """Broadcast features to the return tensor's ensemble and batch dimensions."""
            return jnp.broadcast_to(x, (*lead_shape, x.shape[-1]))

        time_feat = embed_time(times, self.time_embedding, self.time_embed_dim)
        inputs = [returns, broadcast_lead(time_feat), broadcast_lead(observations)]
        if actions is not None:
            inputs.append(broadcast_lead(actions))
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)

        return v


class ActorVectorField(nn.Module):
    """Action vector field with optional raw or sinusoidal time features."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    time_embedding: str = 'sinusoidal'
    time_embed_dim: int = 16
    mlp_class: Any = MLP

    def setup(self) -> None:
        """Initialize the action vector field network."""
        if self.mlp_class is ResidualMLP:
            self.mlp = self.mlp_class(self.hidden_dims, self.action_dim, layer_norm=self.layer_norm)
        else:
            self.mlp = self.mlp_class(
                (*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm
            )

    @nn.compact
    def __call__(self, observations, actions, times=None):
        """Predict action velocities from observations, actions, and optional flow times."""
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            time_feat = embed_time(times, self.time_embedding, self.time_embed_dim)
            inputs = jnp.concatenate([observations, actions, time_feat], axis=-1)

        v = self.mlp(inputs)

        return v


class ImplicitQuantileValue(nn.Module):
    """Implicit quantile V(s) or Q(s, a) network.

    Embedding networks are shared; only the output network is ensembled.
    """

    hidden_dims: Sequence[int]
    tau_embedding_num_cosines: int
    embedding_dim: int
    layer_norm: bool = False
    num_ensembles: int = 2

    def setup(self) -> None:
        """Initialize shared quantile and state-action embeddings and the output network."""
        mlp_class = MLP
        self.tau_cosine_embedding = CosineEmbedding(num_cosines=self.tau_embedding_num_cosines)
        self.tau_embedding_net = mlp_class((self.embedding_dim,), activate_final=False, layer_norm=self.layer_norm)
        self.sa_embedding_net = mlp_class((self.embedding_dim,), activate_final=False, layer_norm=self.layer_norm)

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)
        self.value_net = value_net

    @nn.compact
    def __call__(self, observations, actions=None, taus=None):
        """Predict values at the requested quantile fractions, optionally conditioned on actions."""
        if taus is None:
            raise ValueError('ImplicitQuantileValue requires `taus`.')
        tau_cosine_embeddings = self.tau_cosine_embedding(taus)
        tau_embeddings = self.tau_embedding_net(tau_cosine_embeddings)
        if actions is None:
            inputs = observations
        else:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        sa_embeddings = self.sa_embedding_net(inputs)
        # (*batch, num_taus, embedding_dim)
        embeddings = jnp.expand_dims(sa_embeddings, axis=-2) * tau_embeddings
        # (num_ensembles, *batch, num_taus, 1), without the leading axis when num_ensembles == 1.
        quantiles = self.value_net(embeddings)

        return quantiles


class FloQValueVectorField(nn.Module):
    """Upstream FLOQ critic vector field with HL-Gauss return embeddings."""

    hidden_dims: Sequence[int]
    layer_norm: bool = False
    num_ensembles: int = 1
    output_dim: int = 1

    embed_time: bool = True

    clip_return_embed: bool = False
    q_min: float = 0.0
    q_max: float = 0.0
    sigma: float = 16.0
    num_bins: int = 51

    @staticmethod
    def to_probs(
        target: jax.Array,
        support: jax.Array,
        sigma: float,
    ) -> jax.Array:
        """Embed returns as normalized HL-Gauss bin probabilities, following upstream FLOQ."""
        assert target.ndim == 1

        cdf_evals = jax.scipy.special.erf((support - target[:, None]) / (jnp.sqrt(2.0) * sigma))
        z = cdf_evals[..., -1:] - cdf_evals[..., :1]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z

    @staticmethod
    def compute_support(q_min, q_max, num_bins):
        """Return evenly spaced support points, following upstream FLOQ."""
        return q_min + jnp.arange(num_bins) * (q_max - q_min) / (num_bins - 1)

    def setup(self) -> None:
        """Initialize the FLOQ critic ensemble with separate inputs per member."""
        mlp_class = ensemblize(MLP, self.num_ensembles, in_axes=0)
        self.mlp = mlp_class(
            (*self.hidden_dims, self.output_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    @nn.compact
    def __call__(self, observations, actions, returns, times):
        """Predict return velocities with shape (num_ensembles, batch_size, output_dim)."""

        observations = jnp.concatenate([observations, actions], axis=-1)
        observations = jnp.expand_dims(observations, axis=0)
        observations = jnp.tile(observations, [self.num_ensembles, 1, 1])

        support = self.compute_support(self.q_min, self.q_max, self.num_bins)
        bin_width = support[1] - support[0]
        num_ensembles, batch_size, _ = returns.shape
        returns = jnp.reshape(returns, (num_ensembles * batch_size,))
        if self.clip_return_embed:
            # Avoid 0/0 when both endpoint CDFs saturate outside the finite support.
            returns = jnp.clip(returns, self.q_min, self.q_max)
        returns = self.to_probs(returns, support, self.sigma * bin_width)
        returns = jnp.reshape(returns, (self.num_ensembles, batch_size, self.num_bins - 1))

        if self.embed_time:
            times_embed = jnp.cos(times)
        else:
            times_embed = times
        times_embed = jnp.expand_dims(times_embed, axis=0)
        times_embed = jnp.tile(times_embed, [self.num_ensembles, 1, 1])

        inputs = jnp.concatenate([observations, returns, times_embed], axis=-1)
        return self.mlp(inputs)


class UnconditionalEmbedding(nn.Module):
    """Learned null actions for reading the critic as a state-value flow.

    Use one independent embedding per return-flow ensemble member.
    """

    embed_dim: int
    num_ensembles: int = 1

    @nn.compact
    def __call__(self):
        """Return learned null actions with shape (num_ensembles, embed_dim)."""
        embed = nn.Embed(self.num_ensembles, self.embed_dim)
        return embed(jnp.arange(self.num_ensembles, dtype=jnp.uint8))
