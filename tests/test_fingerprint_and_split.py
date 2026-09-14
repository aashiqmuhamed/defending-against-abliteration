"""Diagnostic outputs, config suggestion bounds, and the XSTest dev split."""

from __future__ import annotations

import pytest
import torch

from ddo_defense.fingerprint import (
    keyword_refusal,
    rotate_direction,
    suggest_config,
)
from ddo_eval.benchmarks.xstest import (
    XSTEST_SAFE_CATEGORIES,
    dev_report_split,
)


def _fp(**kw):
    base = {
        "zone_start": 8,
        "zone_coverage": 0.5,
        "gate_arch": "swiglu",
        # suggest_config reads zone_start, zone_coverage, gate_arch and the
        # layer count; the rest are carried so the double looks like a real
        # fingerprint rather than the minimum that happens to work.
        "rfa_compliance": 0.8,
        "rfa_degenerate": 0.0,
        "d0_to_d60_ratio": 2.0,
        "architecture": {"n_layers": 32},
    }
    base.update(kw)
    return base


def test_keyword_refusal_detects_and_ignores_empties():
    assert keyword_refusal("I cannot help with that") == 1
    assert keyword_refusal("Sure, here is how") == 0
    assert keyword_refusal("") == 0


def test_rotation_has_the_requested_angle(unit_direction):
    for degrees in (30.0, 60.0, 90.0):
        rotated = rotate_direction(unit_direction, degrees, seed=0)
        cos = torch.dot(rotated, unit_direction / unit_direction.norm())
        assert cos.item() == pytest.approx(
            torch.cos(torch.tensor(degrees * torch.pi / 180)).item(), abs=1e-4
        )
        assert rotated.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_suggested_band_is_inside_the_model():
    for zone_start in (0, 3, 8, 20, 31):
        for coverage in (0.1, 0.5, 0.9):
            out = suggest_config(_fp(zone_start=zone_start, zone_coverage=coverage))
            assert 0 <= out["layer_start"] < out["layer_end"] <= 32
            lo, hi = out["search_layer_start"]
            assert 0 <= lo <= hi <= 32, "the search band must fit the model too"
            elo, ehi = out["search_layer_end"]
            assert 0 <= elo <= ehi <= 32


def test_geglu_is_suggested_additive():
    out = suggest_config(_fp(gate_arch="geglu"))
    assert out["compile_mode"] == "additive"
    assert any("GeGLU" in n for n in out["notes"])


def test_swiglu_defaults_to_replace():
    assert suggest_config(_fp())["compile_mode"] == "replace"


def test_no_fitted_defense_family_is_returned():
    out = suggest_config(_fp(rfa_compliance=0.1, zone_coverage=0.02))
    assert "defense_family" not in out


def test_the_upstream_band_reaches_the_layer_before_the_zone():
    """layer_end is exclusive, so it should equal the zone's first layer.

    Writing it inclusive left the band two layers short of the causal zone: with
    a zone starting at 8 the band covered 2..6 and layer 7, which is still
    upstream, was never defended.
    """
    out = suggest_config(_fp(zone_start=8, zone_coverage=0.5))
    assert (out["layer_start"], out["layer_end"]) == (2, 8)
    assert list(range(out["layer_start"], out["layer_end"]))[-1] == 7
    assert out["search_layer_start"] == (1, 4)
    assert out["search_layer_end"] == (6, 12)
    assert out["compile_mode"] == "replace"


def _synthetic_safe_prompts():
    return [
        {"prompt": f"{cat} prompt {i}", "category": cat, "id": f"{cat}-{i}"}
        for cat in XSTEST_SAFE_CATEGORIES
        for i in range(25)
    ]


def test_split_is_stratified_and_disjoint():
    prompts = _synthetic_safe_prompts()
    dev, report = dev_report_split(prompts, dev_size=50)

    assert len(dev) == 50
    assert len(report) == 200

    dev_texts = {p["prompt"] for p in dev}
    report_texts = {p["prompt"] for p in report}
    assert not (dev_texts & report_texts)

    from collections import Counter

    counts = Counter(p["category"] for p in dev)
    assert set(counts) == set(XSTEST_SAFE_CATEGORIES)
    assert set(counts.values()) == {5}


def test_split_is_stable_across_input_order():
    """Membership must not depend on row order, or it drifts between users."""
    import random

    prompts = _synthetic_safe_prompts()
    shuffled = prompts[:]
    random.Random(7).shuffle(shuffled)

    first = {p["prompt"] for p in dev_report_split(prompts, dev_size=50)[0]}
    second = {p["prompt"] for p in dev_report_split(shuffled, dev_size=50)[0]}
    assert first == second


def test_dev_size_zero_reports_everything():
    prompts = _synthetic_safe_prompts()
    dev, report = dev_report_split(prompts, dev_size=0)
    assert dev == []
    assert len(report) == 250


def test_split_scales_with_dev_size():
    prompts = _synthetic_safe_prompts()
    for dev_size, per_cat in ((20, 2), (30, 3), (100, 10)):
        dev, report = dev_report_split(prompts, dev_size=dev_size)
        assert len(dev) == per_cat * len(XSTEST_SAFE_CATEGORIES)
        assert len(dev) + len(report) == 250


def test_negative_dev_size_is_rejected():
    with pytest.raises(ValueError):
        dev_report_split(_synthetic_safe_prompts(), dev_size=-1)

# --- the suggested band must be usable --------------------------------------
# These bounds test the existing heuristic, not a universal claim about layer 0.

def test_suggested_band_never_starts_at_layer_zero():
    for zone_start in range(0, 14):
        for coverage in (0.05, 0.3, 0.7, 0.95):
            out = suggest_config(_fp(zone_start=zone_start, zone_coverage=coverage))
            assert out["layer_start"] >= 1, (zone_start, coverage, out)
            assert out["search_layer_start"][0] >= 1, (zone_start, coverage, out)


def test_suggested_band_is_non_empty_at_any_depth():
    for depth in (2, 3, 4, 8, 12, 32, 42, 80):
        for zone_start in (0, 1, 3, 6, 11):
            fp = _fp(zone_start=zone_start, zone_coverage=0.3)
            fp["architecture"] = {"n_layers": depth}
            out = suggest_config(fp)
            assert 1 <= out["layer_start"] < out["layer_end"] <= depth, (depth, out)


def test_distributed_band_also_avoids_layer_zero():
    fp = _fp(zone_start=0, zone_coverage=0.95)
    fp["architecture"] = {"n_layers": 4}
    out = suggest_config(fp)
    assert out["layer_start"] >= 1
    assert out["layer_start"] < out["layer_end"]


def test_a_suggested_band_can_actually_be_applied(tiny_model, unit_directions_per_layer):
    """End to end: the suggestion must not produce a band the surgery rejects."""
    from ddo_defense.defense.surgery import apply_ddo_to_model

    fp = _fp(zone_start=1, zone_coverage=0.2)
    fp["architecture"] = {"n_layers": tiny_model.config.num_hidden_layers}
    out = suggest_config(fp)

    band = list(range(out["layer_start"], out["layer_end"]))
    assert band, out
    info = apply_ddo_to_model(
        tiny_model, r_by_layer=unit_directions_per_layer, target_layers=band,
        n_decoys=1, beta=2.0, decoy_scale=0.5, seed=0,
        compile_mode=out["compile_mode"],
    )
    assert info["target_layers"] == band


def test_a_fingerprint_without_a_layer_count_is_refused():
    """Every bound in a suggested band is a fraction of the model's depth.

    Guessing 32 returns a plausible-looking band for a model that may not exist:
    on a 48-layer Yi or an 80-layer checkpoint the band would be wrong with
    nothing in the output saying the depth was assumed.
    """
    fp = {
        "zone_start": 8, "zone_coverage": 0.4, "gate_arch": "swiglu",
        "rfa_compliance": 0.8, "rfa_degenerate": 0.0, "d0_to_d60_ratio": 2.0,
    }
    with pytest.raises(ValueError, match="no layer count"):
        suggest_config(fp)


def test_a_layer_count_from_either_place_is_accepted():
    base = {
        "zone_start": 8, "zone_coverage": 0.4, "gate_arch": "swiglu",
        "rfa_compliance": 0.8, "rfa_degenerate": 0.0, "d0_to_d60_ratio": 2.0,
    }
    from_arch = suggest_config({**base, "architecture": {"n_layers": 48}})
    from_zone = suggest_config({**base, "causal_zone": {"n_layers": 48}})
    assert from_arch["search_layer_end"][1] <= 48
    assert from_zone["search_layer_end"][1] <= 48


def test_the_band_scales_with_depth():
    """A 48-layer model must not get a 32-layer band."""
    base = {
        "zone_start": 20, "zone_coverage": 0.9, "gate_arch": "swiglu",
        "rfa_compliance": 0.8, "rfa_degenerate": 0.0, "d0_to_d60_ratio": 2.0,
    }
    small = suggest_config({**base, "architecture": {"n_layers": 32}})
    large = suggest_config({**base, "architecture": {"n_layers": 64}})
    assert large["search_layer_end"][1] > small["search_layer_end"][1]
