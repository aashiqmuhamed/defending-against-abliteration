"""The adaptive multi-phase attack, which had no test coverage at all.

It produces the per-phase curve that the strongest claim in the attack ladder
rests on, and two of its steps -- the degenerate-layer guard and the
Gram-Schmidt orthogonalisation -- are the kind of linear algebra that fails
silently rather than loudly.
"""

from __future__ import annotations

import pytest
import torch

from ddo_defense.attacks.adaptive import (
    _ablation_hooks_for,
    n_phase_attack,
    substring_asr,
)


def test_substring_asr_counts_only_coherent_non_refusals():
    assert substring_asr([
        {"prompt": "p", "response": "Sure, here are the detailed steps you asked for."},
        {"prompt": "p", "response": "I cannot help with that request."},
        {"prompt": "p", "response": ""},
    ]) == pytest.approx(1 / 3)


def test_substring_asr_on_nothing_is_zero_not_an_error():
    assert substring_asr([]) == 0.0


def test_hooks_skip_a_zero_direction(tiny_adapter, unit_directions_per_layer):
    """A zero row means the layer carries no signal, so nothing is ablated there."""
    dirs = torch.stack([torch.as_tensor(d).float() for d in unit_directions_per_layer])
    dirs[0] = 0.0

    pre, post = _ablation_hooks_for(tiny_adapter, [dirs])
    hooked_modules = {id(m) for m, _fn in pre} | {id(m) for m, _fn in post}

    assert id(tiny_adapter.blocks[0]) not in hooked_modules
    assert id(tiny_adapter.blocks[1]) in hooked_modules


def test_saturation_must_be_a_fraction():
    with pytest.raises(ValueError, match="not a fraction"):
        n_phase_attack(
            adapter=None, eval_prompts=["p"], saturation=65,
            harmful_instructions=["h"], harmless_instructions=["s"],
        )


def test_the_result_reports_determinism_rather_than_a_seed(tiny_adapter):
    """Probe selection is a deterministic prefix, so there is no seed to record."""
    out = n_phase_attack(
        tiny_adapter, eval_prompts=["p1", "p2"], max_phases=1, n_train=2,
        harmful_instructions=["a", "bb"], harmless_instructions=["w", "xx"],
        max_new_tokens=4, batch_size=2, verbose=False,
    )
    assert out["deterministic"] is True
    assert "seed" not in out


def test_two_identical_runs_agree(tiny_adapter):
    kwargs = dict(
        eval_prompts=["p1", "p2"], max_phases=2, n_train=2,
        harmful_instructions=["a", "bb"], harmless_instructions=["w", "xx"],
        max_new_tokens=4, batch_size=2, verbose=False,
    )
    a = n_phase_attack(tiny_adapter, **kwargs)
    b = n_phase_attack(tiny_adapter, **kwargs)
    assert a["asr_by_phase"] == b["asr_by_phase"]


def test_phases_that_did_not_run_are_absent_from_the_curve(tiny_adapter):
    """A fabricated value is indistinguishable from a measurement."""
    out = n_phase_attack(
        tiny_adapter, eval_prompts=["p1"], max_phases=3, n_train=2,
        harmful_instructions=["a", "bb"], harmless_instructions=["w", "xx"],
        max_new_tokens=4, batch_size=1, verbose=False,
        asr_fn=lambda c: 1.0,          # saturates immediately
        saturation=0.98,
    )
    assert out["saturated"] is True
    assert out["n_phases_run"] == 1
    assert set(out["asr_by_phase"]) == {1}, "later phases must not be filled in"
