"""The fingerprint, run end to end on a tiny model.

The pure rulebook functions are covered elsewhere. What matters here is that the
measurement path actually executes and returns all five features, because that
path drives generation once per layer and is the part most likely to break
against a real model.
"""

from __future__ import annotations

import pytest
import torch

from ddo_defense.fingerprint import (
    causal_zone,
    detect_architecture,
    fingerprint,
    reader_cone_ratio,
    rfa_compliance,
    suggest_config,
)

FAST = {"max_new_tokens": 3, "batch_size": 2}
PROBES = ["alpha", "beta"]


def test_causal_zone_returns_one_reading_per_layer(tiny_adapter, unit_directions_per_layer):
    zone = causal_zone(
        tiny_adapter, PROBES, unit_directions_per_layer, verbose=False, **FAST
    )
    assert len(zone["per_layer_refusal"]) == tiny_adapter.n_layers
    assert len(zone["per_layer_drop"]) == tiny_adapter.n_layers
    assert zone["n_layers"] == tiny_adapter.n_layers


def test_causal_zone_reports_both_coverage_conventions(tiny_adapter, unit_directions_per_layer):
    """The two conventions are easy to confuse, so both are named and returned."""
    zone = causal_zone(
        tiny_adapter, PROBES, unit_directions_per_layer, verbose=False, **FAST
    )
    assert 0.0 <= zone["zone_coverage"] <= 1.0
    assert 0.0 <= zone["zone_span_fraction"] <= 1.0
    assert 0.0 <= zone["causal_fraction"] <= 1.0


def test_default_threshold_is_the_absolute_ten_percent_drop(
    tiny_adapter, unit_directions_per_layer
):
    """A layer is causal when ablating it alone costs more than 10% of refusal."""
    zone = causal_zone(
        tiny_adapter, PROBES, unit_directions_per_layer, verbose=False, **FAST
    )
    assert zone["drop_threshold"] == pytest.approx(0.10)


def test_a_relative_bar_can_be_opted_into(tiny_adapter, unit_directions_per_layer):
    """A relative bar scaled by baseline refusal is opt-in; the default is absolute."""
    zone = causal_zone(
        tiny_adapter, PROBES, unit_directions_per_layer, verbose=False,
        min_drop=0.1, relative_drop=0.4, **FAST
    )
    baseline = zone["baseline_refusal"]
    assert zone["drop_threshold"] == pytest.approx(max(0.1, 0.4 * baseline))


def test_zone_start_is_in_range_or_signals_absence(tiny_adapter, unit_directions_per_layer):
    zone = causal_zone(
        tiny_adapter, PROBES, unit_directions_per_layer, verbose=False, **FAST
    )
    # Either a real layer index, or n_layers meaning no causal layer was found.
    assert 0 <= zone["zone_start"] <= tiny_adapter.n_layers


def test_reader_cone_ratio_reports_no_signal_as_no_measurement(
    tiny_adapter, unit_directions_per_layer
):
    """A random model's refusal rate does not move, and that is not a breadth.

    Reporting 1.0 when neither the true nor the rotated direction changes refusal
    would be indistinguishable from the documented signature of a broad reader,
    which is a substantive finding about the model.
    """
    cone = reader_cone_ratio(
        tiny_adapter, PROBES, unit_directions_per_layer, **FAST
    )
    assert "d0_to_d60_ratio" in cone
    ratio = cone["d0_to_d60_ratio"]
    if cone["drop_at_0deg"] <= 1e-6 and abs(cone["drop_at_60deg"]) <= 1e-6:
        assert ratio != ratio, "no signal must be reported as nan, not as 1.0"
    else:
        assert ratio == ratio


def test_rfa_compliance_partitions_every_response(tiny_adapter, unit_directions_per_layer):
    """Refusal, degeneracy and compliance must account for all of it."""
    comp = rfa_compliance(
        tiny_adapter, PROBES, unit_directions_per_layer,
        max_new_tokens=3, batch_size=2,
    )
    total = comp["rfa_compliance"] + comp["rfa_degenerate"] + comp["rfa_refusal"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_detect_architecture_reads_the_config(tiny_adapter, tiny_model):
    arch = detect_architecture(tiny_adapter)
    assert arch["gate_arch"] == "swiglu"
    assert arch["n_layers"] == tiny_model.config.num_hidden_layers
    assert arch["d_model"] == tiny_model.config.hidden_size
    assert arch["intermediate_size"] == tiny_model.config.intermediate_size


def test_fingerprint_returns_all_five_features(tiny_adapter, unit_directions_per_layer):
    fp = fingerprint(
        tiny_adapter, PROBES, unit_directions_per_layer,
        n_harmful=2, max_new_tokens=3, batch_size=2, verbose=False,
    )
    for feature in ("zone_start", "zone_coverage", "gate_arch",
                    "rfa_compliance", "d0_to_d60_ratio"):
        assert feature in fp, feature
    assert "defense_family" not in fp


def test_fingerprint_output_feeds_suggest_config(tiny_adapter, unit_directions_per_layer):
    """The two halves must actually compose, not just exist."""
    fp = fingerprint(
        tiny_adapter, PROBES, unit_directions_per_layer,
        n_harmful=2, max_new_tokens=3, batch_size=2, verbose=False,
    )
    suggestion = suggest_config(fp)
    depth = tiny_adapter.n_layers
    assert 0 <= suggestion["layer_start"] < suggestion["layer_end"] <= depth
    assert suggestion["compile_mode"] in ("replace", "additive")


def test_fingerprint_needs_directions_or_harmless_prompts(tiny_adapter):
    with pytest.raises(ValueError, match="harmless_prompts"):
        fingerprint(tiny_adapter, PROBES, None, verbose=False)


def test_fingerprint_estimates_directions_when_not_given(tiny_adapter):
    fp = fingerprint(
        tiny_adapter, PROBES, None, harmless_prompts=["x", "y"],
        n_harmful=2, max_new_tokens=3, batch_size=2, verbose=False,
    )
    assert "zone_start" in fp


def test_geglu_model_is_suggested_additive_end_to_end(tiny_gelu_adapter):
    torch.manual_seed(3)
    dirs = []
    for _ in range(tiny_gelu_adapter.n_layers):
        v = torch.randn(tiny_gelu_adapter.d_model)
        dirs.append(v / v.norm())

    fp = fingerprint(
        tiny_gelu_adapter, PROBES, dirs,
        n_harmful=2, max_new_tokens=3, batch_size=2, verbose=False,
    )
    assert fp["gate_arch"] == "geglu"
    assert suggest_config(fp)["compile_mode"] == "additive"
