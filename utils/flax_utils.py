import functools
import glob
import os
import pickle
import re
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """Named modules with shared parameter storage."""

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Call a named module, or initialize all modules from their keyword argument bundles.

        With name=None, supply one argument bundle per module in kwargs.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Model parameters, optimizer state, and training step with immutable updates."""

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Initialize the training state and optional optimizer state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Apply the model using stored parameters or an explicit parameter override.

        Pass the loss function's traced parameters to differentiate with respect to them.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Bind the name of a module in a ModuleDict."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply gradients and return the updated training state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn, log_grad_stats=False):
        """Differentiate a loss returning (loss, info), then return the updated state and metrics.

        Use a Python bool for log_grad_stats to select whether gradient statistics are computed.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        if log_grad_stats is False:
            return self.apply_gradients(grads=grads), info
        else:
            grad_max = jax.tree_util.tree_map(jnp.max, grads)
            grad_min = jax.tree_util.tree_map(jnp.min, grads)
            grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

            grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
            grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
            grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

            grad_info = {
                'grad/max': jnp.max(grad_max_flat),
                'grad/min': jnp.min(grad_min_flat),
                'grad/norm': jnp.linalg.norm(grad_norm_flat, ord=2),
            }

            info.update(grad_info)
        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, epoch):
    """Atomically save the agent state to params_<epoch>.pkl."""
    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    # The temporary file is excluded from pruning and replaced only after a complete write.
    tmp_path = save_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(save_dict, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, save_path)

    print(f'Saved to {save_path}')


def prune_checkpoints(save_dirs, keep_last):
    """Keep the highest-step checkpoints across the supplied directories without descending into subdirectories.

    Earlier directories win step ties; pass the current run directory first.
    None or -1 disables pruning; 0 removes all matching checkpoints.
    """
    if keep_last is None or keep_last == -1:
        return
    if keep_last < -1:
        raise ValueError('keep_last must be -1 or nonnegative.')
    if isinstance(save_dirs, (str, os.PathLike)):
        save_dirs = [save_dirs]

    # Deduplicate files when directories repeat or resolve to the same location.
    checkpoints = {}
    for rank, save_dir in enumerate(save_dirs):
        for path in glob.glob(os.path.join(glob.escape(os.fspath(save_dir)), 'params_*.pkl')):
            match = re.fullmatch(r'params_(\d+)\.pkl', os.path.basename(path))
            if match is not None:
                checkpoints.setdefault(os.path.realpath(path), (int(match.group(1)), -rank, path))

    for _, _, path in sorted(checkpoints.values(), reverse=True)[keep_last:]:
        try:
            os.remove(path)
        except OSError as exc:
            # Pruning failures must not interrupt training.
            print(f'Could not prune {path}: {exc}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore an agent checkpoint from the single directory matching restore_path."""
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
