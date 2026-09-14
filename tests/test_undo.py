"""The undo attack: can an attacker find and switch off the injected neurons?

Each heuristic exploits a different signature of the injection, so each is tested
against weights built to carry that signature.
"""

from __future__ import annotations

import pytest
import torch

from ddo_defense.attacks.undo import (
    HEURISTICS,
    score_neurons,
    undo_attack,
    undo_layer,
)
from ddo_defense.defense.surgery import apply_ddo_to_layer
from ddo_defense.mlp import get_glu_handles


def test_gate_proj_heuristic_ranks_the_injected_neuron_first(tiny_model, unit_direction):
    """An injected gate row is written as beta*r, so it aligns with r strongly."""
    layer = tiny_model.model.layers[0]
    apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=1, beta=8.0, decoy_scale=0.5,
        neuron_indices=[7], compile_mode="replace", seed=0,
    )
    scores = score_neurons(layer, unit_direction, heuristic="gate_proj")
    assert int(scores.argmax()) == 7


def test_gate_up_cos_heuristic_finds_collinear_rows(tiny_model, unit_direction):
    """Injection writes the same trigger into gate and up, up to a factor."""
    layer = tiny_model.model.layers[0]
    apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=1, beta=3.0, decoy_scale=0.5,
        neuron_indices=[5], compile_mode="replace", seed=0,
    )
    scores = score_neurons(layer, heuristic="gate_up_cos")
    assert int(scores.argmax()) == 5
    assert scores[5] == pytest.approx(1.0, abs=1e-3)


def test_down_norm_heuristic_flags_a_large_output_vector(tiny_model, unit_direction):
    layer = tiny_model.model.layers[0]
    _up, _gate, down = get_glu_handles(layer)
    with torch.no_grad():
        down.weight.data *= 0.01  # make the injected column stand out
    apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=1, beta=2.0, decoy_scale=5.0,
        neuron_indices=[3], compile_mode="replace", seed=0,
    )
    scores = score_neurons(layer, heuristic="down_norm")
    assert int(scores.argmax()) == 3


def test_gate_proj_heuristic_requires_a_direction(tiny_model):
    with pytest.raises(ValueError, match="refusal direction"):
        score_neurons(tiny_model.model.layers[0], None, heuristic="gate_proj")


def test_unknown_heuristic_is_rejected(tiny_model, unit_direction):
    with pytest.raises(ValueError, match="Unknown heuristic"):
        score_neurons(tiny_model.model.layers[0], unit_direction, heuristic="nope")


def test_undo_layer_zeroes_exactly_the_selected_columns(tiny_model, unit_direction):
    layer = tiny_model.model.layers[0]
    _up, _gate, down = get_glu_handles(layer)
    before = down.weight.data.clone()

    ablated = undo_layer(layer, unit_direction, m=3, heuristic="gate_proj")

    assert len(ablated) == 3
    after = down.weight.data
    for idx in range(after.shape[1]):
        if idx in ablated:
            assert torch.all(after[:, idx] == 0)
        else:
            assert torch.allclose(after[:, idx], before[:, idx])


def test_undo_respects_the_budget(tiny_model, unit_direction):
    for m in (1, 2, 8):
        model = tiny_model
        ablated = undo_layer(model.model.layers[1], unit_direction, m=m,
                             heuristic="down_norm")
        assert len(ablated) == m


def test_budget_larger_than_the_layer_is_clamped(tiny_model, unit_direction):
    layer = tiny_model.model.layers[0]
    width = layer.mlp.down_proj.weight.shape[1]
    ablated = undo_layer(layer, unit_direction, m=width + 50, heuristic="down_norm")
    assert len(ablated) == width


def test_zero_budget_ablates_nothing(tiny_model, unit_direction):
    layer = tiny_model.model.layers[0]
    before = layer.mlp.down_proj.weight.data.clone()
    assert undo_layer(layer, unit_direction, m=0, heuristic="down_norm") == []
    assert torch.allclose(layer.mlp.down_proj.weight.data, before)


def test_undo_attack_covers_requested_layers(tiny_model, unit_directions_per_layer):
    info = undo_attack(
        tiny_model, r_by_layer=unit_directions_per_layer,
        target_layers=[0, 1], m=2, heuristic="gate_proj",
    )
    assert info["heuristic"] == "gate_proj"
    assert set(info["ablated"]) == {0, 1}
    assert all(len(v) == 2 for v in info["ablated"].values())


def test_undo_attack_defaults_to_every_layer(tiny_model, unit_directions_per_layer):
    info = undo_attack(tiny_model, r_by_layer=unit_directions_per_layer, m=1)
    assert len(info["ablated"]) == tiny_model.config.num_hidden_layers


def test_undo_attack_needs_directions_for_gate_proj(tiny_model):
    with pytest.raises(ValueError, match="r_by_layer"):
        undo_attack(tiny_model, m=1, heuristic="gate_proj")


def test_undo_works_on_a_fused_model(tiny_fused_model, unit_direction):
    """The fused layout must not hide the neurons from the attack."""
    layer = tiny_fused_model.model.layers[0]
    ablated = undo_layer(layer, unit_direction, m=2, heuristic="gate_up_cos")
    assert len(ablated) == 2
    for idx in ablated:
        assert torch.all(layer.mlp.down_proj.weight.data[:, idx] == 0)


def test_all_heuristics_are_reachable(tiny_model, unit_direction):
    for heuristic in HEURISTICS:
        scores = score_neurons(tiny_model.model.layers[0], unit_direction,
                               heuristic=heuristic)
        assert scores.shape[0] == tiny_model.config.intermediate_size


def test_a_degenerate_direction_is_refused(tiny_model):
    """Every neuron would score 0, so topk would zero an arbitrary m of them."""
    from ddo_defense.attacks.undo import score_neurons

    with pytest.raises(ValueError, match="degenerate"):
        score_neurons(
            tiny_model.model.layers[0], heuristic="gate_proj",
            r=torch.zeros(tiny_model.config.hidden_size),
        )
