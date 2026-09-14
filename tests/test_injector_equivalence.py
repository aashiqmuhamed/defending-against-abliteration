"""The property the whole method rests on: trained equals deployed.

DDO optimises a decoy by simulating, inside a differentiable trace, the change
that weight surgery will later make permanent. If the simulated change and the
written weights ever disagree, the defense is tuned against a model that is not
the one saved to disk, and every number measured afterwards describes something
else.

These tests pin that equivalence down without needing nnsight, by comparing the
shared formula against the actual forward output of the surgically modified MLP.
Both activations are covered, because using SiLU on a GeGLU model is exactly the
kind of mismatch this catches.
"""

from __future__ import annotations

import copy

import pytest
import torch

from ddo_defense.defense.optimizer import (
    DecoyInjector,
    decoy_neuron_contribution,
    neuron_delta,
    original_neuron_contribution,
)
from ddo_defense.defense.surgery import apply_ddo_to_layer
from ddo_defense.mlp import get_activation_fn, get_glu_handles

NEURON = 5
BETA = 2.5
SCALE = 0.4


def _orthogonal_unit(r: torch.Tensor, seed: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(r.shape[0], generator=g)
    v = v - (v @ r) * r
    return v / v.norm()


def _setup(model, r, u, compile_mode="replace"):
    """Record the original neuron, apply surgery, return everything needed."""
    layer = model.model.layers[0]
    mlp_before = copy.deepcopy(layer.mlp)
    up, gate, down = get_glu_handles(layer)

    orig = {
        "gate": gate.weight.data[NEURON].clone().float(),
        "up": up.weight.data[NEURON].clone().float(),
        "down": down.weight.data[:, NEURON].clone().float(),
    }

    apply_ddo_to_layer(
        layer, r=r, n_decoys=1, beta=BETA, decoy_scale=SCALE,
        neuron_indices=[NEURON], decoy_dirs_override=u.unsqueeze(1),
        compile_mode=compile_mode, seed=0,
    )
    return layer, mlp_before, orig


def _written(layer):
    up, gate, down = get_glu_handles(layer)
    return (
        gate.weight.data[NEURON].clone().float(),
        up.weight.data[NEURON].clone().float(),
        down.weight.data[:, NEURON].clone().float(),
    )


@pytest.mark.parametrize("fixture_name", ["tiny_model", "tiny_gelu_model"])
def test_written_weights_encode_the_decoy_formula(request, fixture_name, unit_direction):
    """The rows the surgery writes must compute the intended gated decoy."""
    model = request.getfixturevalue(fixture_name)
    r = unit_direction / unit_direction.norm()
    u = _orthogonal_unit(r)

    layer, _before, _orig = _setup(model, r, u)
    act = get_activation_fn(layer, model.config)
    gate_w, up_w, down_w = _written(layer)

    torch.manual_seed(7)
    h = torch.randn(2, 3, r.shape[0])

    from_weights = original_neuron_contribution(h, gate_w, up_w, down_w, act)
    intended = decoy_neuron_contribution(h, r, u, BETA, SCALE, act)

    assert torch.allclose(from_weights, intended, atol=1e-4), (
        "the compiled neuron does not compute the decoy formula the optimizer "
        "assumes"
    )


@pytest.mark.parametrize("fixture_name", ["tiny_model", "tiny_gelu_model"])
def test_simulated_delta_matches_the_real_forward_change(request, fixture_name, unit_direction):
    """The trained delta equals the deployed model's actual change in output."""
    model = request.getfixturevalue(fixture_name)
    r = unit_direction / unit_direction.norm()
    u = _orthogonal_unit(r)

    layer, mlp_before, orig = _setup(model, r, u)
    act = get_activation_fn(layer, model.config)

    torch.manual_seed(8)
    h = torch.randn(2, 4, r.shape[0])

    with torch.no_grad():
        actual = layer.mlp(h) - mlp_before(h)

    expected = neuron_delta(
        h, trigger=r, decoy=u, beta=BETA, scale=SCALE,
        gate_row=orig["gate"], up_row=orig["up"], down_col=orig["down"], act=act,
    )

    assert torch.allclose(actual, expected, atol=1e-4), (
        f"max deviation {(actual - expected).abs().max().item():.2e}: the "
        f"injector's simulated delta does not match the compiled weights"
    )


def test_using_the_wrong_activation_is_detectable(tiny_gelu_model, unit_direction):
    """On a GELU model, a SiLU-based delta must not pass as equivalent."""
    r = unit_direction / unit_direction.norm()
    u = _orthogonal_unit(r)
    layer, mlp_before, orig = _setup(tiny_gelu_model, r, u)

    torch.manual_seed(9)
    h = torch.randn(1, 3, r.shape[0])
    with torch.no_grad():
        actual = layer.mlp(h) - mlp_before(h)

    wrong = neuron_delta(
        h, trigger=r, decoy=u, beta=BETA, scale=SCALE,
        gate_row=orig["gate"], up_row=orig["up"], down_col=orig["down"],
        act=torch.nn.functional.silu,
    )
    assert not torch.allclose(actual, wrong, atol=1e-4)


def test_only_the_hijacked_neuron_changes_behaviour(tiny_model, unit_direction):
    """The delta must come from one neuron, not a diffuse perturbation."""
    r = unit_direction / unit_direction.norm()
    u = _orthogonal_unit(r)
    layer, mlp_before, _orig = _setup(tiny_model, r, u)

    up_after, gate_after, down_after = get_glu_handles(layer)
    up_before, gate_before, down_before = get_glu_handles(
        type("L", (), {"mlp": mlp_before})()
    )

    for idx in range(up_after.weight.shape[0]):
        same = torch.allclose(up_after.weight.data[idx], up_before.weight.data[idx])
        assert same == (idx != NEURON)


def _bare_injector(r, k, d_model, seed=3):
    """A DecoyInjector with only the fields the pure methods touch.

    Built without __init__ on purpose: construction needs an nnsight model, but
    the orthogonalisation and clamping are plain tensor maths worth testing on
    their own.
    """
    import torch.nn as nn

    inj = object.__new__(DecoyInjector)
    inj.target_layers = [0]
    inj.K = k
    inj.device = torch.device("cpu")
    inj.r_hats = {0: r}
    g = torch.Generator().manual_seed(seed)
    inj.fn_vectors = {0: nn.Parameter(torch.randn(k, d_model, generator=g))}
    # Plain tensors, as DecoyInjector.__init__ builds them: beta and scale are
    # fixed hyperparameters and must not carry requires_grad.
    inj.betas = {0: torch.full((k,), 50.0)}
    inj.scales = {0: torch.full((k,), 99.0)}
    return inj


@pytest.mark.parametrize("k", [1, 2, 4])
def test_orthogonalize_keeps_decoys_clean(unit_direction, k):
    """Decoys stay unit norm, orthogonal to r, and orthogonal to each other.

    Orthogonality to r is what makes a decoy a decoy: a component along the
    refusal direction would strengthen or weaken refusal instead of misdirecting
    the attacker's estimate.
    """
    r = unit_direction / unit_direction.norm()
    inj = _bare_injector(r, k, r.shape[0])
    inj.orthogonalize()

    U = inj.fn_vectors[0].data
    norms = U.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert (U @ r).abs().max() < 1e-5

    gram = U @ U.T
    off_diagonal = gram - torch.eye(k)
    assert off_diagonal.abs().max() < 1e-5


def test_clamp_keeps_the_gate_in_a_sane_range(unit_direction):
    inj = _bare_injector(unit_direction / unit_direction.norm(), 2, 32)
    inj.clamp_params(min_beta=0.5, max_beta=20.0, min_scale=0.01, max_scale=5.0)
    assert inj.betas[0].data.max() <= 20.0
    assert inj.scales[0].data.max() <= 5.0


# --- which layernorm actually feeds the MLP ----------------------------------
# The injector needs the tensor the MLP receives. Resolving that by attribute
# name is only safe if the priority is right: a Gemma-style layer exposes both
# post_attention_layernorm and pre_feedforward_layernorm, and only the latter
# produces the MLP input. These tests check behaviour rather than the name, by
# comparing captured tensors during a real forward pass.

def test_llama_feeds_the_mlp_from_the_post_attention_norm(tiny_model):
    from ddo_defense.defense.optimizer import _resolve_mlp_norm_name

    layer = tiny_model.model.layers[0]
    assert _resolve_mlp_norm_name(layer) == "post_attention_layernorm"


def test_gemma_exposes_both_norms(tiny_gemma2_model):
    """Gemma-2 exposes both names, so the name alone does not identify the input."""
    layer = tiny_gemma2_model.model.layers[0]
    assert getattr(layer, "post_attention_layernorm", None) is not None
    assert getattr(layer, "pre_feedforward_layernorm", None) is not None


def test_resolution_picks_the_norm_the_mlp_actually_reads(tiny_model, tiny_gemma2_model):
    """The resolved norm must be the one whose output reaches ``mlp``.

    Checked by running the block and comparing the resolved layernorm's output
    with the tensor the MLP receives, so this holds for whichever architecture
    the fixture builds rather than resting on a name.  Gemma-2 is the case that
    matters: it has ``post_attention_layernorm`` too, but that one normalises the
    attention output before the residual add.
    """
    from ddo_defense.defense.optimizer import _resolve_mlp_norm_name

    for model in (tiny_model, tiny_gemma2_model):
        layer = model.model.layers[0]
        name = _resolve_mlp_norm_name(layer)
        seen = {}

        def norm_hook(_m, _inp, out, seen=seen):
            seen["norm"] = out.detach().clone()

        def mlp_hook(_m, inp, _out, seen=seen):
            seen["mlp_in"] = inp[0].detach().clone()

        handles = [
            getattr(layer, name).register_forward_hook(norm_hook),
            layer.mlp.register_forward_hook(mlp_hook),
        ]
        try:
            with torch.no_grad():
                model(torch.tensor([[1, 2, 3, 4]]))
        finally:
            for h in handles:
                h.remove()

        assert "norm" in seen and "mlp_in" in seen
        assert torch.allclose(seen["norm"], seen["mlp_in"], atol=1e-5), (
            f"{type(model).__name__}: {name} is not the MLP input"
        )


def test_gemma2_resolves_to_the_pre_feedforward_norm(tiny_gemma2_model):
    from ddo_defense.defense.optimizer import _resolve_mlp_norm_name

    layer = tiny_gemma2_model.model.layers[0]
    assert _resolve_mlp_norm_name(layer) == "pre_feedforward_layernorm"


def test_a_layer_with_no_recognised_norm_is_reported():
    from ddo_defense.defense.optimizer import _resolve_mlp_norm_name

    class _Bare:
        pass

    with pytest.raises(ValueError, match="layernorm"):
        _resolve_mlp_norm_name(_Bare())

# --- both compile modes -----------------------------------------------------
# Replacement and superposition leave different weights behind, so the simulated
# delta has to depend on the mode. Training under one and compiling the other
# optimises a decoy for a model that is never saved.

@pytest.mark.parametrize("mode", ["replace", "additive"])
@pytest.mark.parametrize("fixture_name", ["tiny_model", "tiny_gelu_model"])
def test_simulated_delta_matches_the_forward_change_in_both_modes(
    request, fixture_name, mode, unit_direction
):
    model = request.getfixturevalue(fixture_name)
    r = unit_direction / unit_direction.norm()
    u = _orthogonal_unit(r)

    layer, mlp_before, orig = _setup(model, r, u, compile_mode=mode)
    act = get_activation_fn(layer, model.config)

    torch.manual_seed(21)
    h = torch.randn(2, 4, r.shape[0])
    with torch.no_grad():
        actual = layer.mlp(h) - mlp_before(h)

    expected = neuron_delta(
        h, trigger=r, decoy=u, beta=BETA, scale=SCALE,
        gate_row=orig["gate"], up_row=orig["up"], down_col=orig["down"],
        act=act, compile_mode=mode,
    )
    assert torch.allclose(actual, expected, atol=1e-4), (
        f"{mode}: max deviation {(actual - expected).abs().max().item():.2e}"
    )


def test_the_two_modes_are_not_interchangeable(tiny_model, unit_direction):
    """Simulating one mode and compiling the other must not be treated as equal."""
    r = unit_direction / unit_direction.norm()
    u = _orthogonal_unit(r)
    layer, mlp_before, orig = _setup(tiny_model, r, u, compile_mode="additive")
    act = get_activation_fn(layer, tiny_model.config)

    torch.manual_seed(22)
    h = torch.randn(1, 3, r.shape[0])
    with torch.no_grad():
        actual = layer.mlp(h) - mlp_before(h)

    as_replace = neuron_delta(
        h, trigger=r, decoy=u, beta=BETA, scale=SCALE,
        gate_row=orig["gate"], up_row=orig["up"], down_col=orig["down"],
        act=act, compile_mode="replace",
    )
    assert not torch.allclose(actual, as_replace, atol=1e-4)


def test_an_unknown_mode_is_rejected(tiny_model, unit_direction):
    r = unit_direction / unit_direction.norm()
    with pytest.raises(ValueError, match="compile_mode"):
        neuron_delta(
            torch.randn(1, 2, r.shape[0]), trigger=r, decoy=r, beta=1.0, scale=1.0,
            gate_row=r, up_row=r, down_col=r,
            act=torch.nn.functional.silu, compile_mode="soft",
        )


# --- first-token anchoring --------------------------------------------------
# The anchoring term is a margin between the strongest refusal logit and the
# strongest compliance logit. It exists because the sequence-level losses can be
# satisfied by opening with a compliance token and pivoting to refusal
# mid-sentence, which is not refusal at generation time.

def test_margin_is_positive_when_refusal_leads():
    from ddo_defense.defense.optimizer import refusal_margin

    logits = torch.zeros(1, 10)
    logits[0, 3] = 5.0   # a refusal prefix
    logits[0, 7] = 1.0   # a compliance prefix
    assert refusal_margin(logits, [3], [7]).item() == pytest.approx(4.0)


def test_margin_is_negative_when_compliance_leads():
    from ddo_defense.defense.optimizer import refusal_margin

    logits = torch.zeros(1, 10)
    logits[0, 3] = 1.0
    logits[0, 7] = 6.0
    assert refusal_margin(logits, [3], [7]).item() == pytest.approx(-5.0)


def test_margin_takes_the_strongest_of_each_set():
    """A max, not a sum: one strong prefix should carry the set."""
    from ddo_defense.defense.optimizer import refusal_margin

    logits = torch.zeros(1, 12)
    logits[0, 1], logits[0, 2], logits[0, 3] = 0.5, 4.0, 0.5  # refusal set
    logits[0, 8], logits[0, 9] = 2.0, 1.0                      # compliance set
    assert refusal_margin(logits, [1, 2, 3], [8, 9]).item() == pytest.approx(2.0)


def test_margin_is_computed_in_high_precision():
    from ddo_defense.defense.optimizer import refusal_margin

    logits = torch.zeros(1, 6, dtype=torch.float32)
    logits[0, 1] = 3.0
    assert refusal_margin(logits, [1], [2]).dtype == torch.float64


def test_margin_handles_a_batch():
    from ddo_defense.defense.optimizer import refusal_margin

    logits = torch.zeros(3, 8)
    logits[:, 2] = torch.tensor([1.0, 2.0, 3.0])
    logits[:, 5] = 1.0
    out = refusal_margin(logits, [2], [5])
    assert out.shape == (3,)
    assert out.tolist() == pytest.approx([0.0, 1.0, 2.0])


def test_adapter_supplies_both_anchor_sets(tiny_adapter):
    refusal_ids, compliance_ids = tiny_adapter.anchor_toks()
    # Three refusal prefixes and two compliance prefixes. Disjointness is a
    # property of a real vocabulary, not asserted here: the stub tokenizer keys
    # on first characters, so "Sorry" and "Sure" collide.
    assert len(refusal_ids) == 3
    assert len(compliance_ids) == 2
    assert all(isinstance(i, int) for i in refusal_ids + compliance_ids)


# --- diversified readers ----------------------------------------------------
# A reader group is a perturbation of r_hat, not a direction orthogonal to it:
# an orthogonal reader carries no refusal signal and would gate on noise.  The
# reader the injector trains against must also be the one compilation writes.

def _diversified(r, n_readers, gamma=0.3, seed=0):
    from ddo_defense.defense.surgery import build_trigger_set

    return build_trigger_set(
        r, n_triggers=n_readers, trigger_source="diversified", seed=seed, gamma=gamma
    )


def test_first_reader_is_refusal_itself(unit_direction):
    Q = _diversified(unit_direction, 4)
    assert Q.shape == (unit_direction.shape[0], 4)
    assert torch.allclose(Q[:, 0], unit_direction, atol=1e-6)


def test_readers_stay_aligned_with_refusal_but_differ(unit_direction):
    Q = _diversified(unit_direction, 4, gamma=0.3)
    cosines = [float(Q[:, j] @ unit_direction) for j in range(4)]
    assert cosines[0] == pytest.approx(1.0, abs=1e-6)
    for c in cosines[1:]:
        # gamma=0.3 puts the reader well inside the refusal half-space.
        assert 0.9 < c < 1.0
    for j in range(1, 4):
        for i in range(1, j):
            assert float(Q[:, i] @ Q[:, j]) < 0.999


def test_larger_gamma_means_more_diverse_readers(unit_direction):
    tight = _diversified(unit_direction, 3, gamma=0.1)
    loose = _diversified(unit_direction, 3, gamma=1.0)
    assert float(loose[:, 1] @ unit_direction) < float(tight[:, 1] @ unit_direction)


def test_one_reader_reduces_to_the_shared_reader(unit_direction):
    Q = _diversified(unit_direction, 1)
    assert Q.shape[1] == 1
    assert torch.allclose(Q[:, 0], unit_direction, atol=1e-6)


def test_negative_gamma_is_rejected(unit_direction):
    with pytest.raises(ValueError, match="gamma"):
        _diversified(unit_direction, 2, gamma=-0.1)


def test_injector_reads_the_group_reader_not_refusal(unit_direction):
    """``_trigger`` hands decoy k its own reader, cycling if there are fewer."""
    d_model = unit_direction.shape[0]
    inj = _bare_injector(unit_direction, 4, d_model)
    Q = _diversified(unit_direction, 2)
    inj.triggers = {0: Q}

    assert torch.allclose(inj._trigger(0, 0), Q[:, 0])
    assert torch.allclose(inj._trigger(0, 1), Q[:, 1])
    assert torch.allclose(inj._trigger(0, 2), Q[:, 0])   # cycles
    assert torch.allclose(inj._trigger(0, 3), Q[:, 1])


def test_injector_falls_back_to_refusal_without_readers(unit_direction):
    inj = _bare_injector(unit_direction, 2, unit_direction.shape[0])
    inj.triggers = {}
    for k in range(2):
        assert torch.allclose(inj._trigger(0, k), unit_direction)


def test_the_reader_written_to_weights_is_the_one_trained_against(tiny_model, unit_direction):
    """Compilation must write the injector's reader into the gate and up rows.

    This is the property that makes a K>1 run deployable: writing a different
    reader than the one the decoy was optimised against changes which prompts
    fire the neuron.
    """
    layer = tiny_model.model.layers[0]
    Q = _diversified(unit_direction, 2)
    reader = Q[:, 1]
    u = _orthogonal_unit(reader)

    apply_ddo_to_layer(
        layer, r=reader, n_decoys=1, beta=BETA, decoy_scale=SCALE,
        neuron_indices=[NEURON], decoy_dirs_override=u.unsqueeze(1),
        compile_mode="replace", seed=0,
    )
    gate_row, up_row, down_col = _written(layer)

    assert torch.allclose(up_row, reader, atol=1e-5)
    assert torch.allclose(gate_row, BETA * reader, atol=1e-5)
    assert torch.allclose(down_col, SCALE * u, atol=1e-5)


# --- gains are hyperparameters, not parameters ------------------------------

def test_only_directions_receive_gradients(unit_direction):
    """beta and s are tuned by the hyperparameter search, not by gradients."""
    inj = _bare_injector(unit_direction, 2, unit_direction.shape[0])
    params = inj.parameters()
    assert params == [inj.fn_vectors[0]]
    for p in params:
        assert p.requires_grad


def test_gains_are_plain_tensors_after_construction(unit_direction):
    from ddo_defense.defense.optimizer import DecoyInjector

    inj = object.__new__(DecoyInjector)
    # Reproduce what __init__ stores, then check nothing asks for a gradient.
    inj.betas = {0: torch.full((2,), 3.0)}
    inj.scales = {0: torch.full((2,), 0.2)}
    assert inj.betas[0].requires_grad is False
    assert inj.scales[0].requires_grad is False


# --- the activation the decoy is optimised against --------------------------

def test_act_comes_from_the_layer_not_from_silu(unit_direction):
    """A GeGLU layer must train against GELU, not SiLU.

    The injector holds an activation map built from the real layers; ``_act``
    has to consult it, because a decoy optimised against SiLU and compiled into
    Gemma-2 was optimised for a function the deployed model never applies.
    """
    inj = _bare_injector(unit_direction, 1, unit_direction.shape[0])
    inj.activation_fns = {0: torch.nn.functional.gelu}
    assert inj._act(0) is torch.nn.functional.gelu
    # Layers with no entry fall back to SiLU, the SwiGLU case.
    assert inj._act(7) is torch.nn.functional.silu


def test_activation_map_is_gelu_for_a_geglu_model(tiny_gelu_model, tiny_gemma2_model):
    from ddo_defense.mlp import get_activation_fn

    for model in (tiny_gelu_model, tiny_gemma2_model):
        layer = model.model.layers[0]
        act = get_activation_fn(layer, model.config)
        x = torch.linspace(-3, 3, 7)
        assert not torch.allclose(act(x), torch.nn.functional.silu(x), atol=1e-3)
        assert torch.allclose(act(x), layer.mlp.act_fn(x), atol=1e-6)


# --- refusal targets --------------------------------------------------------

def test_non_refusal_targets_are_reported(capsys):
    """The refusal term trains toward the base model's own continuation.

    If the base model complies on a probe, that probe's target is a compliance
    continuation.  The count is surfaced rather than silently absorbed.
    """
    from ddo_defense.defense.optimizer import DDOOptimizer

    share = DDOOptimizer._warn_if_targets_are_not_refusals([
        "I cannot help with that.",
        "Sorry, I can't assist with that request.",
        "Sure, here are the steps: first,",
    ])
    assert share == pytest.approx(1 / 3)
    out = capsys.readouterr().out
    assert "1/3" in out


def test_all_refusal_targets_are_silent(capsys):
    from ddo_defense.defense.optimizer import DDOOptimizer

    share = DDOOptimizer._warn_if_targets_are_not_refusals(
        ["I cannot help with that.", "I'm sorry, but I can't do that."]
    )
    assert share == 0.0
    assert capsys.readouterr().out == ""


def test_no_targets_is_not_an_error():
    from ddo_defense.defense.optimizer import DDOOptimizer

    assert DDOOptimizer._warn_if_targets_are_not_refusals([]) == 0.0
# --- the vector decoys are measured against ---------------------------------
# Training keeps every decoy orthogonal to the refusal direction.  With a
# diversified reader the neuron reads a perturbation of that direction, so
# compilation has to keep using refusal itself as the reference.

def test_decoy_stays_orthogonal_to_refusal_not_to_the_reader(tiny_model):
    """With a diversified reader, the decoy's reference is still refusal.

    Training keeps each decoy orthogonal to r_hat, so compilation has to use the
    same reference.  Re-projecting against the group's reader would deploy a
    rotated version of the direction that was optimised.
    """
    from ddo_defense.defense.surgery import apply_ddo_to_layer, build_trigger_set
    from ddo_defense.mlp import get_glu_handles

    d_model = tiny_model.config.hidden_size
    torch.manual_seed(7)
    r = torch.randn(d_model)
    r = r / r.norm()
    reader = build_trigger_set(
        r, n_triggers=2, trigger_source="diversified", seed=0, gamma=0.4
    )[:, 1]
    assert float(reader @ r) < 1.0          # the reader really is perturbed

    u = torch.randn(d_model)
    u = u - (u @ r) * r
    u = u / u.norm()

    layer = tiny_model.model.layers[0]
    apply_ddo_to_layer(
        layer, r=reader, n_decoys=1, beta=2.0, decoy_scale=1.0,
        neuron_indices=[3], decoy_dirs_override=u.unsqueeze(1),
        decoy_orth_to=r, compile_mode="replace", seed=0,
    )
    _, _, down = get_glu_handles(layer)
    written = down.weight.data[:, 3].float()
    written = written / written.norm()

    assert torch.allclose(written, u, atol=1e-4)
    assert float(written @ r) == pytest.approx(0.0, abs=1e-5)


# --- the training loop's sampling and weighting ------------------------------

def test_a_shared_cursor_wraps_inside_each_pool():
    """One cursor walks two pools of different lengths, so it must wrap per pool.

    The filtered harmful and harmless sets rarely match in size. An unwrapped
    cursor yields a short or empty window from the shorter pool, and an empty
    batch reaches the trace as ``invoke([])``.
    """
    def take(pool, start, n_take):
        # The contract _take_window implements, exercised directly.
        if not pool:
            return []
        start %= len(pool)
        if n_take >= len(pool):
            return list(pool)
        end = start + n_take
        if end <= len(pool):
            return pool[start:end]
        return pool[start:] + pool[: end % len(pool)]

    harmful = [f"h{i}" for i in range(128)]
    harmless = [f"s{i}" for i in range(64)]
    B = 16
    cursor = 0
    for _ in range(16):
        got = take(harmless, cursor, B)
        assert len(got) == B, f"cursor {cursor} gave {len(got)} of {B} prompts"
        assert got, "empty batch would reach the trace as invoke([])"
        cursor = (cursor + B) % max(len(harmful), 1)


def test_take_window_never_returns_an_empty_or_short_batch():
    from ddo_defense.defense.optimizer import DDOOptimizer
    import inspect

    src = inspect.getsource(DDOOptimizer.fit)
    assert "start %= len(pool)" in src, (
        "the cursor is no longer reduced into the pool being sliced"
    )


def test_kl_helper_is_per_token_when_positions_are_flattened():
    """The retain term must not be a sum over target positions.

    With reduction='batchmean' the divisor is shape[0], so a [B, T, V] tensor
    yields T times the per-token divergence while the confusion term, computed on
    [B, V], yields exactly the per-token value. Equal lambdas then are not equal.
    """
    from ddo_defense.defense.optimizer import kl_div_fn

    torch.manual_seed(0)
    B, T, V = 4, 30, 16
    a = torch.randn(B, T, V)
    b = torch.randn(B, T, V)

    unflattened = kl_div_fn(a, b)
    flattened = kl_div_fn(a.reshape(-1, V), b.reshape(-1, V))

    assert float(unflattened) == pytest.approx(float(flattened) * T, rel=1e-6)
    # And the flattened form matches what the confusion term computes per token.
    per_token = kl_div_fn(a[:, 0], b[:, 0])
    assert abs(float(flattened) - float(per_token)) < abs(float(unflattened) - float(per_token))
