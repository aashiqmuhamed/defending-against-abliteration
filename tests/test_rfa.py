"""The RFA attack path, exercised on a tiny model.

Rank and probe budget are parameters rather than separate entry points, so both
are checked, along with the three ablation surfaces.
"""

from __future__ import annotations

import os

import pytest
import torch

from ddo_defense.attacks.rfa import (
    ABLATION_MODES,
    RFA_VARIANTS,
    build_ablation_hooks,
    compute_attack_directions,
    compute_mean_harmless_projection,
    generate_under_attack,
    run_rfa,
    run_rfa_variant,
)


def _dirs(adapter, rank=1):
    torch.manual_seed(4)
    out = torch.zeros(adapter.n_layers, rank, adapter.d_model)
    for layer in range(adapter.n_layers):
        for k in range(rank):
            v = torch.randn(adapter.d_model)
            out[layer, k] = v / v.norm()
    return out


def test_three_point_hooks_cover_three_surfaces(tiny_adapter):
    dirs = _dirs(tiny_adapter)
    pre, post = build_ablation_hooks(tiny_adapter, dirs, ablation_mode="three_point")
    n = tiny_adapter.n_layers
    assert len(pre) == n           # block input
    assert len(post) == 2 * n      # attention output and MLP output


def test_residual_stream_hooks_cover_only_the_block_input(tiny_adapter):
    dirs = _dirs(tiny_adapter)
    pre, post = build_ablation_hooks(tiny_adapter, dirs, ablation_mode="residual_stream")
    assert len(pre) == tiny_adapter.n_layers
    assert post == []


def test_higher_rank_adds_more_hooks(tiny_adapter):
    pre1, _ = build_ablation_hooks(tiny_adapter, _dirs(tiny_adapter, 1),
                                   ablation_mode="residual_stream")
    pre4, _ = build_ablation_hooks(tiny_adapter, _dirs(tiny_adapter, 4),
                                   ablation_mode="residual_stream")
    assert len(pre4) == 4 * len(pre1)


def test_zero_directions_are_skipped(tiny_adapter):
    """A padded direction tensor must be safe to pass."""
    dirs = _dirs(tiny_adapter, rank=2)
    dirs[:, 1] = 0.0
    pre, _ = build_ablation_hooks(tiny_adapter, dirs, ablation_mode="residual_stream")
    assert len(pre) == tiny_adapter.n_layers


def test_unknown_ablation_mode_is_rejected(tiny_adapter):
    with pytest.raises(ValueError, match="ablation_mode"):
        build_ablation_hooks(tiny_adapter, _dirs(tiny_adapter), ablation_mode="nope")


def test_patched_mode_requires_the_harmless_mean(tiny_adapter):
    with pytest.raises(ValueError, match="mean_harmless"):
        build_ablation_hooks(tiny_adapter, _dirs(tiny_adapter),
                             ablation_mode="residual_stream_patched")


def test_patched_mode_adds_the_harmless_projection(tiny_adapter):
    dirs = _dirs(tiny_adapter)
    mean_harmless = compute_mean_harmless_projection(tiny_adapter, ["a", "b"], dirs)
    assert mean_harmless.shape == (tiny_adapter.n_layers, tiny_adapter.d_model)
    pre, _ = build_ablation_hooks(
        tiny_adapter, dirs, ablation_mode="residual_stream_patched",
        mean_harmless=mean_harmless,
    )
    # One ablation plus one addition per layer.
    assert len(pre) == 2 * tiny_adapter.n_layers


def test_every_declared_mode_builds(tiny_adapter):
    dirs = _dirs(tiny_adapter)
    mean_harmless = compute_mean_harmless_projection(tiny_adapter, ["a"], dirs)
    for mode in ABLATION_MODES:
        kwargs = {"mean_harmless": mean_harmless} if "patched" in mode else {}
        build_ablation_hooks(tiny_adapter, dirs, ablation_mode=mode, **kwargs)


def test_attack_directions_have_the_requested_rank(tiny_adapter):
    dirs = compute_attack_directions(
        tiny_adapter, rank=2, n_probes=4, filter_probes=False,
        harmful_instructions=["a", "bb", "ccc", "dddd"],
        harmless_instructions=["w", "xx", "yyy", "zzzz"],
        verbose=False,
    )
    assert dirs.shape == (tiny_adapter.n_layers, 2, tiny_adapter.d_model)


def test_probe_budget_caps_the_prompts_used(tiny_adapter):
    """The budget must actually truncate, and the count used must be reported.

    Asserting the returned shape proves nothing here: it is the layer count,
    which holds however many probes were consumed.
    """
    stats = {}
    compute_attack_directions(
        tiny_adapter, rank=1, n_probes=2, filter_probes=False,
        harmful_instructions=["a", "bb", "ccc", "dddd"],
        harmless_instructions=["w", "xx", "yyy", "zzzz"],
        verbose=False, stats=stats,
    )
    assert stats["n_probes_used"] == 2, (
        f"asked for 2 probes from a pool of 4 but used {stats.get('n_probes_used')}"
    )


def test_a_budget_larger_than_the_pool_reports_what_was_available(tiny_adapter):
    """Reporting a budget the attack did not have understates the defense."""
    stats = {}
    compute_attack_directions(
        tiny_adapter, rank=1, n_probes=99, filter_probes=False,
        harmful_instructions=["a", "bb", "ccc"],
        harmless_instructions=["w", "xx", "yyy"],
        verbose=False, stats=stats,
    )
    assert stats["n_probes_used"] == 3


def test_filtering_never_empties_a_side(tiny_adapter):
    """An empty side would make the mean difference meaningless.

    The guard is the `or` in `kept_h or harmful_instructions`, so what has to be
    checked is that a side which filters to nothing falls back to the unfiltered
    prompts rather than proceeding with zero.
    """
    from ddo_defense.directions import filter_probes_by_refusal

    harmful = ["a", "bb", "ccc"]
    harmless = ["w", "xx", "yyy"]
    kept_h, kept_s = filter_probes_by_refusal(tiny_adapter, harmful, harmless)

    assert kept_h, "the harmful side filtered to nothing and was not restored"
    assert kept_s, "the harmless side filtered to nothing and was not restored"
    # A random model has no refusal behaviour, so at least one side is expected to
    # filter empty and fall back to its full set.
    assert kept_h == harmful or kept_s == harmless

    dirs = compute_attack_directions(
        tiny_adapter, rank=1, n_probes=3, filter_probes=True,
        harmful_instructions=harmful, harmless_instructions=harmless,
        verbose=False,
    )
    assert torch.isfinite(dirs).all()
    # Same contract as the estimators: unit norm, or exactly zero for a layer
    # that carries no signal. Layer 0 is zero because every prompt ends with the
    # same template suffix.
    norms = dirs.norm(dim=-1).flatten().tolist()
    for n in norms:
        assert n == pytest.approx(0.0, abs=1e-8) or n == pytest.approx(1.0, abs=1e-3)
    assert any(n > 0.5 for n in norms), "no layer produced a usable direction"


def test_generation_under_attack_returns_one_record_per_prompt(tiny_adapter):
    dataset = [{"instruction": f"prompt {i}"} for i in range(4)]
    out = generate_under_attack(
        tiny_adapter, dataset, _dirs(tiny_adapter),
        max_new_tokens=3, batch_size=2, show_progress=False,
    )
    assert len(out) == 4
    assert all(set(r) == {"prompt", "response"} for r in out)
    assert [r["prompt"] for r in out] == [d["instruction"] for d in dataset]


def test_no_directions_means_no_attack(tiny_adapter):
    dataset = [{"instruction": "p"}]
    out = generate_under_attack(
        tiny_adapter, dataset, None, max_new_tokens=3, show_progress=False
    )
    assert len(out) == 1


def test_the_four_variants_are_declared():
    assert set(RFA_VARIANTS) == {
        "RFA", "RFA_residual", "RFA_harmbench", "RFA_residual_harmbench",
    }
    modes = {v["ablation_mode"] for v in RFA_VARIANTS.values()}
    sets = {v["eval_set"] for v in RFA_VARIANTS.values()}
    assert modes == {"three_point", "residual_stream"}
    assert sets == {"jailbreakbench", "harmbench_standard"}


def test_unknown_variant_is_rejected(tiny_adapter):
    with pytest.raises(ValueError, match="Unknown variant"):
        run_rfa_variant(tiny_adapter, "RFA_nonexistent")

# --- probe accounting -------------------------------------------------------
# A finite prompt pool silently caps the requested budget. An attack that reports
# a budget it did not actually have understates the defense, so the counts that
# were really used are part of the result.

PROMPT_CACHE = "/tmp/ddo_datacheck"


def test_stats_record_the_counts_actually_used(tiny_adapter):
    stats = {}
    compute_attack_directions(
        tiny_adapter, rank=1, n_probes=99, filter_probes=False,
        harmful_instructions=["a", "bb", "ccc"],
        harmless_instructions=["w", "xx", "yyy"],
        verbose=False, stats=stats,
    )
    # 99 was requested; only three of each exist.
    assert stats["n_harmful_used"] == 3
    assert stats["n_harmless_used"] == 3
    assert stats["n_probes_used"] == 3


def test_stats_take_the_smaller_side(tiny_adapter):
    """The usable budget is bounded by whichever pool is shorter."""
    stats = {}
    compute_attack_directions(
        tiny_adapter, rank=1, n_probes=10, filter_probes=False,
        harmful_instructions=["a", "bb"],
        harmless_instructions=["w", "xx", "yyy", "zzzz"],
        verbose=False, stats=stats,
    )
    assert stats["n_harmful_used"] == 2
    assert stats["n_harmless_used"] == 4
    assert stats["n_probes_used"] == 2


def test_stats_are_optional(tiny_adapter):
    dirs = compute_attack_directions(
        tiny_adapter, rank=1, n_probes=2, filter_probes=False,
        harmful_instructions=["a", "bb"], harmless_instructions=["w", "xx"],
        verbose=False,
    )
    assert dirs.shape[0] == tiny_adapter.n_layers


@pytest.mark.skipif(
    not os.path.isdir(PROMPT_CACHE),
    reason="prompt cache not populated; run ddo_defense.data.prefetch first",
)
def test_run_rfa_end_to_end_reports_both_budgets(tiny_adapter, monkeypatch):
    """The whole attack entry point, on real prompts from the verified cache."""
    monkeypatch.setenv("DDO_CACHE_DIR", PROMPT_CACHE)

    eval_data = [{"instruction": f"prompt {i}"} for i in range(3)]
    result = run_rfa(
        tiny_adapter, rank=1, n_probes=300, filter_probes=False,
        eval_data=eval_data, max_new_tokens=2, batch_size=3, verbose=False,
    )

    assert result["attack"] == "RFA"
    assert result["n_probes_requested"] == 300
    # The harmful split is smaller than 300, so the real budget is lower.
    assert result["n_probes_used"] < 300
    assert result["n_probes_used"] == min(
        result["n_harmful_probes"], result["n_harmless_probes"]
    )
    assert result["n_completions"] == 3
    assert len(result["completions"]) == 3


def test_a_saturation_given_as_a_percentage_is_refused():
    """The curve is reported in percent but compared as a fraction.

    Passing 65 meaning 65% would make the comparison `asr >= 65` and early
    stopping could never fire, silently running every phase.
    """
    from ddo_defense.attacks.adaptive import n_phase_attack

    with pytest.raises(ValueError, match="not a fraction"):
        n_phase_attack(
            adapter=None, eval_prompts=["p"], saturation=65,
            harmful_instructions=["h"], harmless_instructions=["s"],
        )
