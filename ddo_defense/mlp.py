"""MLP handle discovery for gated-linear-unit (GLU) feedforward blocks.

DDO repurposes individual MLP neurons, so it needs write access to three
weight matrices per layer:

* ``gate`` -- rows of the gating projection, written with ``beta * trigger``
* ``up``   -- rows of the value projection, written with ``trigger``
* ``down`` -- columns of the output projection, written with the decoy vector

Most GLU models expose these as three separate ``nn.Linear`` modules
(``gate_proj``, ``up_proj``, ``down_proj``).  Some, notably GLM-4, fuse the
first two into a single ``gate_up_proj`` whose rows are the gate half followed
by the up half.  :func:`get_glu_handles` normalises both layouts to the same
three-handle interface, returning lightweight views for the fused case so that
writes propagate back into the real parameter.

The activation must also come from the layer rather than being assumed.
SwiGLU models use SiLU, but GeGLU models such as Gemma-2 use
``gelu_pytorch_tanh``, so a decoy has to be optimised against whichever one the
layer actually applies.  Use :func:`get_activation_fn`.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn.functional as F


class _SlicedParam:
    """A row-slice of a parameter tensor that behaves enough like a parameter.

    Exposes ``data``, ``shape``, ``device``, ``dtype`` and item access.  All
    reads and writes are views into the underlying fused tensor, so in-place
    operations performed through this object mutate the real weight.
    """

    def __init__(self, full_weight: torch.Tensor, start: int, end: int):
        self._full = full_weight
        self._start = start
        self._end = end

    @property
    def data(self) -> torch.Tensor:
        return self._full.data[self._start:self._end]

    @data.setter
    def data(self, value: torch.Tensor) -> None:
        self._full.data[self._start:self._end] = value

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._end - self._start, self._full.shape[1])

    def __getitem__(self, idx):
        return self._full.data[self._start:self._end][idx]

    def __setitem__(self, idx, value) -> None:
        self._full.data[self._start:self._end][idx] = value

    @property
    def device(self) -> torch.device:
        return self._full.device

    @property
    def dtype(self) -> torch.dtype:
        return self._full.dtype


class _FusedProjView:
    """Presents one half of a fused ``gate_up_proj`` as a Linear-like module."""

    def __init__(self, fused_weight: torch.Tensor, start: int, end: int):
        self.weight = _SlicedParam(fused_weight, start, end)


def is_fused_glu(layer_module) -> bool:
    """True when the layer fuses gate and up into a single projection."""
    mlp = getattr(layer_module, "mlp", None)
    if mlp is None:
        return False
    has_split = all(getattr(mlp, n, None) is not None for n in ("gate_proj", "up_proj"))
    return not has_split and getattr(mlp, "gate_up_proj", None) is not None


def get_glu_handles(layer_module) -> Tuple[object, object, object]:
    """Return ``(up, gate, down)`` handles for a layer's GLU MLP.

    Handles the split layout (``gate_proj`` / ``up_proj`` / ``down_proj``) and
    the fused layout (``gate_up_proj`` / ``down_proj``).  For the fused layout
    the first half of the rows is the gate and the second half is the up
    projection, matching the reference implementations in transformers.

    Raises
    ------
    ValueError
        If the layer has no ``.mlp`` or neither layout is recognised.
    """
    mlp = getattr(layer_module, "mlp", None)
    if mlp is None:
        raise ValueError("Layer has no .mlp attribute")

    up = getattr(mlp, "up_proj", None)
    gate = getattr(mlp, "gate_proj", None)
    down = getattr(mlp, "down_proj", None)
    if up is not None and gate is not None and down is not None:
        return up, gate, down

    fused = getattr(mlp, "gate_up_proj", None)
    if fused is not None and down is not None:
        mid = fused.weight.shape[0] // 2
        gate_view = _FusedProjView(fused.weight, 0, mid)
        up_view = _FusedProjView(fused.weight, mid, fused.weight.shape[0])
        return up_view, gate_view, down

    raise ValueError(
        "Expected a GLU MLP exposing either (gate_proj, up_proj, down_proj) "
        "or (gate_up_proj, down_proj)"
    )


def get_intermediate_size(layer_module) -> int:
    """Number of neurons in the layer's MLP, i.e. the gate/up output width."""
    up, _gate, _down = get_glu_handles(layer_module)
    return int(up.weight.shape[0])


def get_activation_fn(layer_module, config=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the activation actually used by this layer's MLP.

    Resolution order:

    1. ``mlp.act_fn`` (current transformers convention)
    2. ``mlp.activation_fn`` / ``mlp.act`` (older or third-party modules)
    3. ``ACT2FN[config.hidden_act]`` when a config is supplied
    4. ``F.silu`` as a last resort

    Falling back to step 4 is only correct for SwiGLU models, so callers that
    care about exactness should pass ``config``.
    """
    mlp = getattr(layer_module, "mlp", None)
    if mlp is not None:
        for attr in ("act_fn", "activation_fn", "act"):
            fn = getattr(mlp, attr, None)
            if callable(fn):
                return fn

    if config is not None:
        name = getattr(config, "hidden_act", None) or getattr(config, "hidden_activation", None)
        if name:
            try:
                from transformers.activations import ACT2FN

                if name in ACT2FN:
                    act = ACT2FN[name]
                    return act() if isinstance(act, type) else act
            except Exception as exc:
                print(f"  warning: could not resolve activation {name!r}: {exc}")
            else:
                if name not in ACT2FN:
                    print(
                        f"  warning: activation {name!r} is not in ACT2FN; "
                        f"falling back to SiLU, which is wrong for a GeGLU model"
                    )

    return F.silu
