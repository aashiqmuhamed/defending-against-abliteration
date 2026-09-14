"""Weight surgery: what gets written, and how the two compile modes differ."""

from __future__ import annotations

import pytest
import torch

from ddo_defense.defense.surgery import (
    apply_ddo_to_layer,
    apply_ddo_to_model,
    build_trigger_set,
    select_neuron_indices,
)
from ddo_defense.mlp import get_glu_handles


def test_replace_writes_exact_trigger_and_decoy(tiny_model, unit_direction):
    layer = tiny_model.model.layers[0]
    up, gate, down = get_glu_handles(layer)
    beta, scale = 3.0, 0.5

    indices, info = apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=1, beta=beta, decoy_scale=scale,
        neuron_indices=[0], compile_mode="replace", seed=0,
    )

    idx = indices[0]
    r_hat = unit_direction / unit_direction.norm()

    assert torch.allclose(up.weight.data[idx], r_hat.to(up.weight.dtype), atol=1e-5)
    assert torch.allclose(
        gate.weight.data[idx], (beta * r_hat).to(gate.weight.dtype), atol=1e-5
    )
    # The decoy column must be orthogonal to the refusal direction, which is the
    # property that makes it a decoy rather than a refusal amplifier.
    written = down.weight.data[:, idx].float()
    assert abs(torch.dot(written / written.norm(), r_hat)) < 1e-3
    assert written.norm() == pytest.approx(scale, abs=1e-3)
    assert info["compile_mode"] == "replace"


def test_additive_equals_original_plus_delta(tiny_model, unit_direction):
    layer = tiny_model.model.layers[0]
    up, gate, down = get_glu_handles(layer)

    before_up = up.weight.data[0].clone()
    before_gate = gate.weight.data[0].clone()
    before_down = down.weight.data[:, 0].clone()

    apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=1, beta=2.0, decoy_scale=0.25,
        neuron_indices=[0], compile_mode="additive", seed=0,
    )

    r_hat = (unit_direction / unit_direction.norm()).to(up.weight.dtype)
    assert torch.allclose(up.weight.data[0], before_up + r_hat, atol=1e-5)
    assert torch.allclose(gate.weight.data[0], before_gate + 2.0 * r_hat, atol=1e-5)
    # The down column moved, and it moved by exactly the decoy vector's norm.
    delta = (down.weight.data[:, 0] - before_down).float()
    assert delta.norm() == pytest.approx(0.25, abs=1e-3)


def test_additive_preserves_more_than_replace(tiny_model, unit_direction):
    """Replace discards the neuron's original function; additive keeps it."""
    import copy

    m_replace = copy.deepcopy(tiny_model)
    m_additive = copy.deepcopy(tiny_model)
    original = tiny_model.model.layers[0].mlp.up_proj.weight.data[0].clone()

    for model, mode in ((m_replace, "replace"), (m_additive, "additive")):
        apply_ddo_to_layer(
            model.model.layers[0], r=unit_direction, n_decoys=1, beta=2.0,
            decoy_scale=0.25, neuron_indices=[0], compile_mode=mode, seed=0,
        )

    replaced = m_replace.model.layers[0].mlp.up_proj.weight.data[0]
    added = m_additive.model.layers[0].mlp.up_proj.weight.data[0]
    assert (replaced - original).norm() > 0
    assert torch.allclose(added - original, replaced, atol=1e-5)


def test_unknown_compile_mode_is_rejected(tiny_model, unit_direction):
    with pytest.raises(ValueError, match="compile_mode"):
        apply_ddo_to_layer(
            tiny_model.model.layers[0], r=unit_direction, n_decoys=1,
            neuron_indices=[0], compile_mode="soft", seed=0,
        )


def test_fused_model_can_be_defended(tiny_fused_model, unit_direction):
    layer = tiny_fused_model.model.layers[0]
    fused_before = layer.mlp.gate_up_proj.weight.data.clone()

    apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=1, beta=2.0, decoy_scale=0.5,
        neuron_indices=[0], compile_mode="replace", seed=0,
    )

    fused_after = layer.mlp.gate_up_proj.weight.data
    mid = fused_after.shape[0] // 2
    assert not torch.allclose(fused_after[0], fused_before[0])
    assert not torch.allclose(fused_after[mid], fused_before[mid])
    # Only the hijacked neuron changed, in both halves.
    assert torch.allclose(fused_after[1:mid], fused_before[1:mid])


def test_low_norm_selection_picks_smallest_columns(tiny_model):
    down = tiny_model.model.layers[0].mlp.down_proj.weight.data
    norms = down.float().norm(dim=0)
    chosen = select_neuron_indices(
        n_neurons=3, intermediate_size=down.shape[1],
        method="low_norm", down_proj_weight=down,
    )
    expected = set(norms.argsort()[:3].tolist())
    assert set(chosen) == expected


def test_multi_decoy_columns_stay_orthogonal(tiny_model, unit_direction):
    """Several decoys in one layer must not collapse onto the same direction."""
    layer = tiny_model.model.layers[0]
    _up, _gate, down = get_glu_handles(layer)

    indices, _ = apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=4, beta=2.0, decoy_scale=1.0,
        compile_mode="replace", seed=0,
    )

    cols = torch.stack([down.weight.data[:, i].float() for i in indices])
    cols = cols / cols.norm(dim=1, keepdim=True)
    gram = cols @ cols.T
    off_diag = gram - torch.eye(len(indices))
    assert off_diag.abs().max() < 1e-2

    r_hat = unit_direction / unit_direction.norm()
    assert (cols @ r_hat).abs().max() < 1e-2


def test_apply_to_model_reports_config(tiny_model):
    r_by_layer = []
    for _ in range(tiny_model.config.num_hidden_layers):
        v = torch.randn(tiny_model.config.hidden_size)
        r_by_layer.append(v / v.norm())

    info = apply_ddo_to_model(
        tiny_model, r_by_layer=r_by_layer, target_layers=[0, 1],
        n_decoys=2, beta=2.0, decoy_scale=0.5, seed=0, compile_mode="additive",
    )
    assert info["compile_mode"] == "additive"
    assert info["target_layers"] == [0, 1]
    assert set(info["neuron_indices"]) == {"0", "1"}


def test_trigger_set_is_unit_norm(unit_direction):
    triggers = build_trigger_set(unit_direction, n_triggers=3,
                                trigger_source="diversified", seed=0)
    assert triggers.shape == (32, 3)
    norms = triggers.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    # Column 0 stays the refusal direction itself.
    assert torch.allclose(triggers[:, 0], unit_direction / unit_direction.norm(), atol=1e-5)


def test_an_unknown_trigger_source_is_rejected(unit_direction):
    with pytest.raises(ValueError, match="trigger_source"):
        build_trigger_set(unit_direction, n_triggers=2, trigger_source="nope")

# --- degenerate directions --------------------------------------------------
# A layer where harmful and harmless activations coincide yields a zero
# direction. Normalising it gives zeros, and writing that as a trigger zeroes the
# neuron's gate and up rows: the neuron goes dead and defends nothing. This is
# reachable whenever a probe set fails to separate at some layer, so the surgery
# refuses rather than writing a dead neuron.

def test_a_zero_direction_is_refused(tiny_model):
    zero = torch.zeros(tiny_model.config.hidden_size)
    with pytest.raises(ValueError, match="degenerate"):
        apply_ddo_to_layer(
            tiny_model.model.layers[0], r=zero, n_decoys=1,
            neuron_indices=[0], compile_mode="replace", seed=0,
        )


def test_refusing_leaves_the_neuron_untouched(tiny_model):
    """The guard must fire before any weight is written."""
    layer = tiny_model.model.layers[0]
    up, gate, down = get_glu_handles(layer)
    before = up.weight.data[0].clone()
    with pytest.raises(ValueError):
        apply_ddo_to_layer(
            layer, r=torch.zeros(tiny_model.config.hidden_size), n_decoys=1,
            neuron_indices=[0], compile_mode="replace", seed=0,
        )
    assert torch.allclose(up.weight.data[0], before)


def test_a_written_trigger_is_never_all_zero(tiny_model, unit_direction):
    layer = tiny_model.model.layers[0]
    up, _gate, _down = get_glu_handles(layer)
    indices, _ = apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=1, beta=2.0, decoy_scale=0.5,
        neuron_indices=[0], compile_mode="replace", seed=0,
    )
    assert up.weight.data[indices[0]].float().norm() > 0.5


def test_model_level_skips_degenerate_layers(tiny_model, unit_directions_per_layer):
    """Layer 0 is dropped, the rest are still defended, and it is reported."""
    dirs = list(unit_directions_per_layer)
    dirs[0] = torch.zeros_like(dirs[0])

    info = apply_ddo_to_model(
        tiny_model, r_by_layer=dirs, target_layers=[0, 1],
        n_decoys=1, beta=2.0, decoy_scale=0.5, seed=0,
    )
    assert info["skipped_layers"] == [0]
    assert info["target_layers"] == [1]
    assert set(info["neuron_indices"]) == {"1"}


def test_model_level_raises_when_every_layer_is_degenerate(tiny_model):
    dirs = [torch.zeros(tiny_model.config.hidden_size)
            for _ in range(tiny_model.config.num_hidden_layers)]
    with pytest.raises(ValueError, match="nothing to build a decoy around"):
        apply_ddo_to_model(
            tiny_model, r_by_layer=dirs, target_layers=[0, 1],
            n_decoys=1, beta=2.0, seed=0,
        )


def test_healthy_layers_report_no_skips(tiny_model, unit_directions_per_layer):
    info = apply_ddo_to_model(
        tiny_model, r_by_layer=unit_directions_per_layer, target_layers=[0, 1],
        n_decoys=1, beta=2.0, decoy_scale=0.5, seed=0,
    )
    assert info["skipped_layers"] == []


def test_neuron_indices_must_match_n_decoys(tiny_model, unit_direction):
    """One decoy direction is built per neuron, so the counts have to agree.

    A mismatch used to write part of the layer and then fail partway through the
    loop, leaving the MLP half-modified.
    """
    layer = tiny_model.model.layers[0]
    before = get_glu_handles(layer)[0].weight.data.clone()

    with pytest.raises(ValueError, match="neuron_indices"):
        apply_ddo_to_layer(
            layer, r=unit_direction, n_decoys=2, beta=2.0, decoy_scale=1.0,
            neuron_indices=[1, 2, 3], compile_mode="replace", seed=0,
        )
    # Refused before touching anything.
    assert torch.equal(get_glu_handles(layer)[0].weight.data, before)


def test_fewer_indices_than_decoys_is_also_refused(tiny_model, unit_direction):
    with pytest.raises(ValueError, match="n_decoys=3"):
        apply_ddo_to_layer(
            layer_module=tiny_model.model.layers[0], r=unit_direction, n_decoys=3,
            beta=2.0, decoy_scale=1.0, neuron_indices=[4], compile_mode="replace",
        )


# --- the multi-reader mechanism ----------------------------------------------
# `triggers[:, j % n_triggers]` is the whole of it: which reader each hijacked
# neuron gets. No test called apply_ddo_to_layer with more than one reader.

def test_readers_are_assigned_round_robin_across_neurons(tiny_model, unit_direction):
    from ddo_defense.defense.surgery import build_trigger_set

    layer = tiny_model.model.layers[0]
    up, gate, down = get_glu_handles(layer)
    indices = [3, 7, 11, 15]

    apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=4, beta=2.0, decoy_scale=1.0,
        neuron_indices=indices, n_triggers=2, trigger_source="diversified",
        reader_gamma=0.4, trigger_seed=5, compile_mode="replace", seed=0,
    )

    Q = build_trigger_set(
        unit_direction, n_triggers=2, trigger_source="diversified",
        seed=5, gamma=0.4,
    )
    for j, idx in enumerate(indices):
        written = up.weight.data[idx].float()
        expected = Q[:, j % 2].float()
        cos = float(written @ expected / (written.norm() * expected.norm() + 1e-9))
        assert cos == pytest.approx(1.0, abs=1e-4), (
            f"neuron {idx} (position {j}) should read reader {j % 2}, cos={cos:.4f}"
        )
        # And the gate row is beta times the same reader.
        assert torch.allclose(gate.weight.data[idx].float(), 2.0 * expected, atol=1e-4)


def test_every_decoy_gets_its_own_direction(tiny_model, unit_direction):
    """Readers cycle, decoy directions do not: each neuron writes a distinct one."""
    layer = tiny_model.model.layers[0]
    _up, _gate, down = get_glu_handles(layer)
    indices = [2, 6, 10, 14]

    apply_ddo_to_layer(
        layer, r=unit_direction, n_decoys=4, beta=2.0, decoy_scale=1.0,
        neuron_indices=indices, n_triggers=2, trigger_source="diversified",
        compile_mode="replace", seed=0,
    )

    cols = [down.weight.data[:, i].float() for i in indices]
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            cos = abs(float(cols[a] @ cols[b] / (cols[a].norm() * cols[b].norm() + 1e-9)))
            assert cos < 0.1, f"decoys {a} and {b} are nearly parallel (cos={cos:.3f})"
