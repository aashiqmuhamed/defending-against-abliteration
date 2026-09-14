"""MLP handle discovery, including the fused layout and activation resolution."""

from __future__ import annotations

import torch

from ddo_defense.mlp import (
    get_activation_fn,
    get_glu_handles,
    get_intermediate_size,
    is_fused_glu,
)


def test_split_layout_returns_three_handles(tiny_model):
    layer = tiny_model.model.layers[0]
    up, gate, down = get_glu_handles(layer)
    assert up.weight.shape == layer.mlp.up_proj.weight.shape
    assert gate.weight.shape == layer.mlp.gate_proj.weight.shape
    assert down.weight.shape == layer.mlp.down_proj.weight.shape
    assert not is_fused_glu(layer)


def test_fused_layout_is_detected(tiny_fused_model):
    layer = tiny_fused_model.model.layers[0]
    assert is_fused_glu(layer)
    up, gate, down = get_glu_handles(layer)
    mid = layer.mlp.gate_up_proj.weight.shape[0] // 2
    assert gate.weight.shape == (mid, layer.mlp.gate_up_proj.weight.shape[1])
    assert up.weight.shape == (mid, layer.mlp.gate_up_proj.weight.shape[1])


def test_fused_writes_land_in_the_right_half(tiny_fused_model):
    """The first half of the fused rows is the gate, the second half is up."""
    layer = tiny_fused_model.model.layers[0]
    up, gate, down = get_glu_handles(layer)
    fused = layer.mlp.gate_up_proj.weight
    mid = fused.shape[0] // 2

    gate_vec = torch.full((fused.shape[1],), 2.0)
    up_vec = torch.full((fused.shape[1],), 3.0)

    with torch.no_grad():
        gate.weight[0] = gate_vec
        up.weight[0] = up_vec

    assert torch.allclose(fused.data[0], gate_vec)
    assert torch.allclose(fused.data[mid], up_vec)


def test_intermediate_size_matches_config(tiny_model):
    assert get_intermediate_size(tiny_model.model.layers[0]) == \
        tiny_model.config.intermediate_size


def test_activation_comes_from_the_layer(tiny_model, tiny_gelu_model):
    """A GeGLU model must not be handed SiLU."""
    x = torch.linspace(-3, 3, 16)

    silu_fn = get_activation_fn(tiny_model.model.layers[0], tiny_model.config)
    gelu_fn = get_activation_fn(tiny_gelu_model.model.layers[0], tiny_gelu_model.config)

    assert torch.allclose(silu_fn(x), torch.nn.functional.silu(x), atol=1e-5)
    assert not torch.allclose(gelu_fn(x), torch.nn.functional.silu(x), atol=1e-3)
