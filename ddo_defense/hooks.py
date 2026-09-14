"""Forward-hook utilities for activation ablation and steering.

The hooks edit the activation in place.  That is what makes them cheap enough to
apply at every layer, but it means the tensor a caller hands in is modified; do
not keep a reference to it expecting the original values.

``add_hooks`` is a context manager that registers forward pre-hooks and forward
hooks for the duration of a block and always removes them afterwards, including
on exception.  The remaining helpers build the individual hook functions used by
Refusal Feature Ablation (RFA) and by activation-addition baselines.

Two ablation surfaces matter:

* *residual_stream* -- project the direction out of each block's input only.
* *three_point* -- project it out of the block input, the attention output and
  the MLP output.  This is the stronger attack, and the default here.

:func:`get_all_direction_ablation_hooks` assembles the three-point variant.

Activations are ``[batch, seq, d_model]`` and directions are ``[d_model]``.
"""

import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from torch import Tensor

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Tensor = input[0]  # [batch, seq, d_model]
        else:
            activation: Tensor = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Tensor = output[0]  # [batch, seq, d_model]
        else:
            activation: Tensor = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_activation_addition_input_pre_hook(vector: Tensor, coeff: Tensor):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Tensor = input[0]  # [batch, seq, d_model]
        else:
            activation: Tensor = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn
