"""Ablation hooks and direction estimation, the shared substrate.

The hooks are what the attack is made of, and the directions are what both the
attack and the defense are built on. Both are checked against hand-computed
answers rather than only for shape.
"""

from __future__ import annotations

import pytest
import torch

from ddo_defense.directions import (
    compute_rank_k_directions,
    estimate_refusal_directions,
    get_mean_activations,
    get_mean_diff,
    get_refusal_scores,
    refusal_score,
)
from ddo_defense.hooks import (
    add_hooks,
    get_activation_addition_input_pre_hook,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)


def test_ablation_removes_the_component_along_the_direction():
    """The defining property: afterwards the projection onto r is zero."""
    torch.manual_seed(0)
    d = torch.randn(16)
    d = d / d.norm()
    x = torch.randn(2, 5, 16)

    hook = get_direction_ablation_input_pre_hook(direction=d.clone())
    out = hook(None, (x.clone(),))[0]

    assert torch.allclose(out @ d, torch.zeros(2, 5), atol=1e-5)


def test_ablation_leaves_orthogonal_content_untouched():
    torch.manual_seed(0)
    d = torch.zeros(8)
    d[0] = 1.0
    x = torch.randn(1, 3, 8)
    x[..., 0] = 5.0

    hook = get_direction_ablation_input_pre_hook(direction=d.clone())
    out = hook(None, (x.clone(),))[0]

    assert torch.allclose(out[..., 1:], x[..., 1:], atol=1e-6)
    assert torch.allclose(out[..., 0], torch.zeros(1, 3), atol=1e-5)


def test_ablation_handles_bare_tensors_and_tuples():
    d = torch.zeros(4)
    d[0] = 1.0
    x = torch.ones(1, 2, 4)

    hook = get_direction_ablation_input_pre_hook(direction=d.clone())
    assert isinstance(hook(None, (x.clone(),)), tuple)
    assert isinstance(hook(None, x.clone()), torch.Tensor)


def test_output_hook_ablates_too():
    d = torch.zeros(4)
    d[1] = 1.0
    x = torch.ones(1, 2, 4)
    hook = get_direction_ablation_output_hook(direction=d.clone())
    out = hook(None, None, (x.clone(),))[0]
    assert torch.allclose(out[..., 1], torch.zeros(1, 2), atol=1e-5)


def test_activation_addition_shifts_by_the_vector():
    v = torch.ones(4)
    x = torch.zeros(1, 2, 4)
    hook = get_activation_addition_input_pre_hook(vector=v, coeff=torch.tensor(2.0))
    out = hook(None, (x.clone(),))[0]
    assert torch.allclose(out, torch.full((1, 2, 4), 2.0), atol=1e-6)


def test_add_hooks_removes_them_afterwards(tiny_model):
    layer = tiny_model.model.layers[0]
    before = len(layer._forward_pre_hooks)

    def noop(module, inp):
        return None

    with add_hooks(module_forward_pre_hooks=[(layer, noop)], module_forward_hooks=[]):
        assert len(layer._forward_pre_hooks) == before + 1
    assert len(layer._forward_pre_hooks) == before


def test_add_hooks_removes_them_after_an_exception(tiny_model):
    layer = tiny_model.model.layers[0]
    before = len(layer._forward_pre_hooks)

    def noop(module, inp):
        return None

    with pytest.raises(RuntimeError):
        with add_hooks(module_forward_pre_hooks=[(layer, noop)], module_forward_hooks=[]):
            raise RuntimeError("boom")
    assert len(layer._forward_pre_hooks) == before


def test_mean_activations_shape_and_precision(tiny_adapter):
    acts = get_mean_activations(
        tiny_adapter.model, ["a", "b", "c"],
        tiny_adapter.tokenize_instructions_fn, tiny_adapter.blocks,
        batch_size=2, positions=(-1,),
    )
    assert acts.shape == (1, tiny_adapter.n_layers, tiny_adapter.d_model)
    # High precision is deliberate: the mean difference of two large sums loses
    # accuracy in bf16.
    assert acts.dtype == torch.float64


def test_mean_diff_is_the_difference_of_means(tiny_adapter):
    harmful, harmless = ["x", "y"], ["p", "q"]
    mh = get_mean_activations(tiny_adapter.model, harmful,
                              tiny_adapter.tokenize_instructions_fn,
                              tiny_adapter.blocks, batch_size=2)
    ms = get_mean_activations(tiny_adapter.model, harmless,
                              tiny_adapter.tokenize_instructions_fn,
                              tiny_adapter.blocks, batch_size=2)
    diff = get_mean_diff(tiny_adapter.model, harmful, harmless,
                         tiny_adapter.tokenize_instructions_fn,
                         tiny_adapter.blocks, batch_size=2)
    assert torch.allclose(diff, mh - ms, atol=1e-10)


def test_estimated_directions_are_unit_or_exactly_zero(tiny_adapter):
    """Every layer is either a unit direction or exactly zero, never in between.

    Layer 0 is legitimately zero: its last-token activation is the embedding of
    the template suffix, which every prompt shares, so harmful and harmless
    means coincide. Zero is the signal that there is nothing to ablate there.
    """
    dirs = estimate_refusal_directions(tiny_adapter, ["a", "bb"], ["c", "dd"])
    assert dirs.shape == (tiny_adapter.n_layers, tiny_adapter.d_model)

    norms = dirs.norm(dim=-1)
    for n in norms.tolist():
        assert n == pytest.approx(0.0, abs=1e-8) or n == pytest.approx(1.0, abs=1e-4)
    assert (norms > 0.5).any(), "no layer produced a usable direction"


def test_both_estimators_agree_on_degenerate_layers(tiny_adapter):
    """The two entry points must not disagree about which layers carry signal."""
    harmful, harmless = ["a", "bb"], ["c", "dd"]
    plain = estimate_refusal_directions(tiny_adapter, harmful, harmless)
    rank1 = compute_rank_k_directions(tiny_adapter, harmful, harmless, k=1)

    plain_zero = (plain.norm(dim=-1) < 1e-8).tolist()
    rank_zero = (rank1[:, 0].norm(dim=-1) < 1e-8).tolist()
    assert plain_zero == rank_zero


def test_rank_one_matches_the_mean_difference(tiny_adapter):
    harmful, harmless = ["a", "bb", "ccc"], ["x", "yy", "zzz"]
    rank1 = compute_rank_k_directions(tiny_adapter, harmful, harmless, k=1)
    plain = estimate_refusal_directions(tiny_adapter, harmful, harmless)
    assert rank1.shape == (tiny_adapter.n_layers, 1, tiny_adapter.d_model)
    compared = 0
    for layer in range(tiny_adapter.n_layers):
        if plain[layer].norm() < 1e-8:
            continue  # degenerate layer, covered by its own test
        cos = torch.dot(rank1[layer, 0].float(), plain[layer].float())
        assert abs(cos.item()) == pytest.approx(1.0, abs=1e-3)
        compared += 1
    assert compared > 0


def test_rank_k_columns_are_unit_norm_or_exactly_zero(tiny_adapter):
    """The same contract the rank-1 estimator holds to, at every rank.

    A column is a direction to ablate or it is nothing.  Zero means "this layer
    carries no signal", and build_ablation_hooks skips it; a rescaled speck of
    numerical noise would have the attack ablate something it has no reason to
    touch.  Layer 0 is always in this state, because its last-token activation is
    the template suffix every prompt shares.
    """
    dirs = compute_rank_k_directions(
        tiny_adapter, ["a", "bb", "ccc", "dddd"], ["w", "xx", "yyy", "zzzz"], k=3
    )
    assert dirs.shape == (tiny_adapter.n_layers, 3, tiny_adapter.d_model)

    for layer in range(tiny_adapter.n_layers):
        for i in range(3):
            n = float(dirs[layer, i].norm())
            assert n == pytest.approx(0.0, abs=1e-8) or n == pytest.approx(1.0, abs=1e-3), (
                f"layer {layer} column {i} has norm {n}, which is neither a "
                f"direction nor absent"
            )
    assert (dirs.norm(dim=-1) > 0.5).any(), "no layer produced a usable direction"


def test_a_degenerate_layer_is_zero_at_every_rank(tiny_adapter):
    """Both estimators must agree about which layers carry signal.

    An SVD of an all-zero contrast returns an orthonormal U with zero singular
    values, so a rank-k attack would otherwise ablate k arbitrary basis
    directions at a layer the rank-1 attack correctly leaves alone.
    """
    harmful, harmless = ["a", "bb", "ccc", "dddd"], ["w", "xx", "yyy", "zzzz"]
    rank1 = estimate_refusal_directions(tiny_adapter, harmful, harmless)
    rank3 = compute_rank_k_directions(tiny_adapter, harmful, harmless, k=3)

    degenerate = [l for l in range(tiny_adapter.n_layers) if rank1[l].norm() < 1e-8]
    assert degenerate, "fixture no longer has a degenerate layer to check"
    for l in degenerate:
        assert float(rank3[l].norm()) == pytest.approx(0.0, abs=1e-8), (
            f"layer {l} is degenerate at rank 1 but carries {rank3[l].norm()} at rank 3"
        )


def test_rank_k_keeps_the_mean_difference_as_its_leading_direction(tiny_adapter):
    """The contrast matrix is uncentered, so the top mode tracks the DIM vector.

    Centering the per-sample contrasts subtracts their mean, which is the DIM
    direction itself, so a rank-k attack would structurally exclude the very
    component the rank-1 attack ablates.
    """
    harmful = ["a", "bb", "ccc", "dddd", "eeeee"]
    harmless = ["v", "ww", "xxx", "yyyy", "zzzzz"]
    dirs = compute_rank_k_directions(tiny_adapter, harmful, harmless, k=3)
    plain = estimate_refusal_directions(tiny_adapter, harmful, harmless)

    compared = 0
    for layer in range(tiny_adapter.n_layers):
        if plain[layer].norm() < 1e-8:
            continue
        cos = abs(float(torch.dot(dirs[layer, 0].float(), plain[layer].float())))
        assert cos > 0.5, (
            f"layer {layer}: leading rank-k direction is nearly orthogonal to the "
            f"mean difference (cos={cos:.3f})"
        )
        compared += 1
    assert compared > 0


def test_rank_must_be_positive(tiny_adapter):
    with pytest.raises(ValueError, match="k must be"):
        compute_rank_k_directions(tiny_adapter, ["a"], ["b"], k=0)


def test_refusal_score_is_log_odds_of_the_refusal_tokens():
    # Two-token vocabulary, all probability on token 0.
    logits = torch.tensor([[[50.0, -50.0]]])
    assert refusal_score(logits, [0]).item() > 0
    assert refusal_score(logits, [1]).item() < 0


def test_refusal_scores_run_over_a_batch(tiny_adapter):
    scores = get_refusal_scores(
        tiny_adapter.model, ["a", "b", "c"],
        tiny_adapter.tokenize_instructions_fn, tiny_adapter.refusal_toks,
        batch_size=2,
    )
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


# --- the estimator the defense builds its trigger from -----------------------
# estimate_refusal_directions_mlp_input is what the decoy's trigger is measured
# from, and it had no test. It must read the layernorm that feeds the MLP, not
# the block input the attacker reads.

def test_mlp_input_estimator_shapes_and_contract(tiny_adapter):
    from ddo_defense.directions import estimate_refusal_directions_mlp_input

    dirs = estimate_refusal_directions_mlp_input(
        tiny_adapter, ["a", "bb", "ccc"], ["w", "xx", "yyy"]
    )
    assert dirs.shape == (tiny_adapter.n_layers, tiny_adapter.d_model)
    for layer in range(tiny_adapter.n_layers):
        n = float(dirs[layer].norm())
        assert n == pytest.approx(0.0, abs=1e-8) or n == pytest.approx(1.0, abs=1e-4), (
            f"layer {layer} has norm {n}, which is neither a direction nor absent"
        )


def test_the_two_estimators_read_different_tensors(tiny_adapter):
    """The attacker reads the residual stream; the decoy reads the MLP input.

    They are deliberately different quantities, so they must not come out equal.
    """
    from ddo_defense.directions import estimate_refusal_directions_mlp_input

    harmful, harmless = ["a", "bb", "ccc"], ["w", "xx", "yyy"]
    residual = estimate_refusal_directions(tiny_adapter, harmful, harmless)
    mlp_input = estimate_refusal_directions_mlp_input(tiny_adapter, harmful, harmless)

    compared = 0
    for layer in range(tiny_adapter.n_layers):
        if residual[layer].norm() < 1e-8 or mlp_input[layer].norm() < 1e-8:
            continue
        assert not torch.allclose(residual[layer], mlp_input[layer], atol=1e-6), (
            f"layer {layer}: both estimators returned the same vector, so one of "
            f"them is reading the wrong tensor"
        )
        compared += 1
    assert compared > 0, "no layer carried signal in both estimators"
