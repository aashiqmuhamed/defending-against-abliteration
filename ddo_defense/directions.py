"""Refusal direction estimation and refusal scoring.

The refusal direction is the difference in mean activations between harmful and
harmless instructions, measured at the last prompt position and taken per layer.
This is the quantity the abliteration attack estimates and projects out, so both
the defense and the attack are built on the functions here.

* :func:`estimate_refusal_directions` reads the block input, the residual stream.
  This is the attacker's surface: one unit vector per layer, which the attack
  ablates.
* :func:`estimate_refusal_directions_mlp_input` reads the output of the layernorm
  that feeds the MLP.  This is the defense's surface, because it is the space the
  gate and up projections read and therefore the space a decoy's trigger has to
  live in.
* :func:`compute_rank_k_directions` returns a ``k``-dimensional subspace per
  layer.  ``k=1`` is the plain mean-difference direction; ``k>1`` takes the
  leading left singular vectors of the per-sample contrast matrix, which is the
  stronger attack.
* :func:`get_refusal_scores` gives a log-odds score of the model starting its
  reply with a refusal token, used to filter probe prompts.

The first two are different tensors on purpose, and the distinction matters:
fitting a decoy against the residual-stream direction would optimise it against a
direction the hijacked neuron never sees.

The attacker recomputes all of this on the *defended* model, so nothing here
depends on having the original weights.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Sequence

import torch
from torch import Tensor
from tqdm import tqdm

from ddo_defense.hooks import add_hooks


# --------------------------------------------------------------------------
# Mean activations
# --------------------------------------------------------------------------

def _mean_activations_pre_hook(layer: int, cache: Tensor, n_samples: int, positions: Sequence[int]):
    def hook_fn(module, inp):
        act = inp[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * act[:, list(positions), :].sum(dim=0)
    return hook_fn


def get_mean_activations(
    model,
    instructions: Sequence[str],
    tokenize_fn,
    block_modules,
    *,
    batch_size: int = 32,
    positions: Sequence[int] = (-1,),
    show_progress: bool = False,
) -> Tensor:
    """Mean block-input activations over ``instructions``.

    Returns ``[n_positions, n_layers, d_model]`` in float64.  High precision is
    deliberate: the mean difference of two large sums loses accuracy in bf16.
    """
    n_positions = len(positions)
    n_layers = int(model.config.num_hidden_layers)
    n_samples = len(instructions)
    d_model = int(model.config.hidden_size)

    cache = torch.zeros(
        (n_positions, n_layers, d_model), dtype=torch.float64, device=model.device
    )
    hooks = [
        (block_modules[layer],
         _mean_activations_pre_hook(layer, cache, n_samples, positions))
        for layer in range(n_layers)
    ]

    rng = range(0, len(instructions), batch_size)
    for i in (tqdm(rng, desc="mean acts") if show_progress else rng):
        enc = tokenize_fn(instructions=list(instructions[i:i + batch_size]))
        with torch.no_grad():
            with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
                model(
                    input_ids=enc.input_ids.to(model.device),
                    attention_mask=enc.attention_mask.to(model.device),
                )
    return cache


def get_mean_diff(
    model,
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    tokenize_fn,
    block_modules,
    *,
    batch_size: int = 32,
    positions: Sequence[int] = (-1,),
) -> Tensor:
    """``mean(harmful) - mean(harmless)``, shape ``[n_positions, n_layers, d_model]``."""
    mh = get_mean_activations(
        model, harmful_instructions, tokenize_fn, block_modules,
        batch_size=batch_size, positions=positions,
    )
    ms = get_mean_activations(
        model, harmless_instructions, tokenize_fn, block_modules,
        batch_size=batch_size, positions=positions,
    )
    return mh - ms


def estimate_refusal_directions(
    adapter,
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    *,
    batch_size: int = 32,
    position: int = -1,
) -> Tensor:
    """Per-layer unit refusal directions, shape ``[n_layers, d_model]``.

    Parameters
    ----------
    adapter
        A :class:`ddo_defense.models.ModelAdapter`.
    harmful_instructions, harmless_instructions
        Raw instruction strings.  128 of each is the usual count.
    position
        Prompt position to read.  ``-1`` is the last token, which is where the
        decision to refuse is legible.
    """
    diff = get_mean_diff(
        adapter.model,
        harmful_instructions,
        harmless_instructions,
        adapter.tokenize_instructions_fn,
        adapter.blocks,
        batch_size=batch_size,
        positions=(position,),
    )[0]
    # A layer with no signal gets exactly zero rather than a rescaled speck.
    # The ablation hooks skip zero directions, which is the honest outcome: at
    # layer 0 the last-token activation depends only on the template suffix,
    # which every prompt shares, so the mean difference is genuinely zero there.
    norms = diff.norm(dim=-1, keepdim=True)
    return torch.where(norms < 1e-10, torch.zeros_like(diff), diff / (norms + 1e-8))


def estimate_refusal_directions_mlp_input(
    adapter,
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    *,
    batch_size: int = 32,
) -> Tensor:
    """Per-layer unit directions read at the MLP input, shape ``[n_layers, d_model]``.

    The defense and the attacker read different tensors, deliberately.  A decoy
    neuron sees the output of the layernorm that feeds the MLP, so that is where
    the direction it is built around must be measured.  An attacker works on the
    residual stream, which is :func:`estimate_refusal_directions`.

    Accumulated in float64, and unfiltered: no refusal-score selection of probes.
    """
    from ddo_defense.defense.optimizer import _resolve_mlp_norm_name

    model = adapter.model
    n_layers = adapter.n_layers
    d_model = adapter.d_model

    def _means(instructions: Sequence[str]) -> Tensor:
        total = torch.zeros((n_layers, d_model), dtype=torch.float64, device=model.device)
        seen = 0
        norms = [
            getattr(adapter.blocks[l], _resolve_mlp_norm_name(adapter.blocks[l]))
            for l in range(n_layers)
        ]

        captured: Dict[int, Tensor] = {}

        def make_hook(layer_idx):
            def hook(_module, _inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                captured[layer_idx] = tensor[:, -1, :].detach()
            return hook

        handles = [norms[l].register_forward_hook(make_hook(l)) for l in range(n_layers)]
        try:
            for i in range(0, len(instructions), batch_size):
                enc = adapter.tokenize_instructions_fn(
                    instructions=list(instructions[i:i + batch_size])
                )
                captured.clear()
                with torch.no_grad():
                    model(
                        input_ids=enc.input_ids.to(model.device),
                        attention_mask=enc.attention_mask.to(model.device),
                    )
                for l in range(n_layers):
                    if l in captured:
                        total[l] += captured[l].to(torch.float64).sum(dim=0)
                seen += enc.input_ids.shape[0]
        finally:
            for h in handles:
                h.remove()

        return total / max(seen, 1)

    diff = _means(list(harmful_instructions)) - _means(list(harmless_instructions))
    norms = diff.norm(dim=-1, keepdim=True)
    return torch.where(norms < 1e-10, torch.zeros_like(diff), diff / (norms + 1e-8))


# --------------------------------------------------------------------------
# Rank-k subspaces
# --------------------------------------------------------------------------

def _collect_last_token_activations(
    model, instructions: Sequence[str], tokenize_fn, block_modules, *, batch_size: int = 8
) -> List[Tensor]:
    """Per-layer ``[n_samples, d_model]`` last-token activations, on CPU."""
    n_layers = int(model.config.num_hidden_layers)
    by_layer: List[List[Tensor]] = [[] for _ in range(n_layers)]

    def hook(storage, layer_idx):
        def hook_fn(module, inp):
            act = inp[0] if isinstance(inp, tuple) else inp
            storage.append((layer_idx, act[:, -1, :].detach().clone().cpu()))
        return hook_fn

    for i in range(0, len(instructions), batch_size):
        enc = tokenize_fn(instructions=list(instructions[i:i + batch_size]))
        storage: List = []
        hooks = [(block_modules[l], hook(storage, l)) for l in range(n_layers)]
        with torch.no_grad():
            with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
                model(
                    input_ids=enc.input_ids.to(model.device),
                    attention_mask=enc.attention_mask.to(model.device),
                )
        for layer_idx, act in storage:
            by_layer[layer_idx].append(act)

    return [torch.cat(chunks, dim=0) if chunks else torch.empty(0) for chunks in by_layer]


def compute_rank_k_directions(
    adapter,
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    *,
    k: int = 1,
    batch_size: int = 16,
) -> Tensor:
    """Per-layer rank-``k`` refusal subspace, shape ``[n_layers, k, d_model]``.

    ``k=1`` is the mean-difference direction, identical to the standard attack.
    For ``k>1`` the columns are the leading left singular vectors of the
    contrast matrix whose columns are the per-sample harmful-minus-harmless
    differences.  Each column is unit norm or exactly zero: a layer with no
    signal stays zero at every rank, and a mode with no strength is left empty
    rather than filled with a rescaled speck of noise.  The SVD makes the
    non-zero columns mutually orthogonal.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    model = adapter.model
    n_layers = adapter.n_layers
    d_model = adapter.d_model

    mean_diff = get_mean_diff(
        model,
        harmful_instructions,
        harmless_instructions,
        adapter.tokenize_instructions_fn,
        adapter.blocks,
        batch_size=batch_size,
        positions=(-1,),
    )[0]

    if torch.isnan(mean_diff).any():
        raise ValueError(
            "Mean activation difference contains NaN. The forward pass produced "
            "NaN, usually a dtype or a broken-checkpoint problem."
        )

    directions = torch.zeros(
        n_layers, k, d_model, device=mean_diff.device, dtype=mean_diff.dtype
    )

    def _unit(v: Tensor) -> Tensor:
        """Unit-normalise, or return zeros when the layer carries no signal.

        Zeros are correct here: :func:`build_ablation_hooks` skips them, so a
        degenerate layer is simply not attacked.  Substituting a random
        direction would make the attack ablate noise it has no reason to remove,
        and would disagree with :func:`estimate_refusal_directions` about the
        same layer.
        """
        n = v.norm()
        if not torch.isfinite(n) or n < 1e-10:
            return torch.zeros_like(v)
        return v / (n + 1e-8)

    if k == 1:
        for layer in range(n_layers):
            directions[layer, 0] = _unit(mean_diff[layer])
        return directions

    harmful_acts = _collect_last_token_activations(
        model, harmful_instructions, adapter.tokenize_instructions_fn,
        adapter.blocks, batch_size=batch_size,
    )
    harmless_acts = _collect_last_token_activations(
        model, harmless_instructions, adapter.tokenize_instructions_fn,
        adapter.blocks, batch_size=batch_size,
    )

    for layer in range(n_layers):
        h, s = harmful_acts[layer], harmless_acts[layer]
        n = min(h.shape[0], s.shape[0])
        fallback = _unit(mean_diff[layer])

        if n == 0:
            directions[layer, :] = fallback
            continue

        diffs = (h[:n] - s[:n]).to(dtype=mean_diff.dtype)

        # A layer with no signal must stay exactly zero at every rank, the same
        # contract estimate_refusal_directions holds to.  An SVD of an all-zero
        # contrast returns an orthonormal U with zero singular values, so without
        # this the rank-k attack would ablate k arbitrary basis directions at a
        # layer the rank-1 attack correctly leaves alone.
        if fallback.norm() < 1e-10:
            continue

        try:
            # The contrast matrix holds the per-sample harmful-minus-harmless
            # differences as columns, uncentered.  Subtracting their mean would
            # remove the mean-difference direction, which is exactly the rank-1
            # DIM component the attack is built around.
            U, S, _ = torch.linalg.svd(diffs.T, full_matrices=False)
            k_actual = min(k, U.shape[1])
            for i in range(k_actual):
                # A mode with no strength is not a direction; leave it at zero so
                # the hooks skip it rather than ablating numerical noise.
                if S[i] <= 1e-10:
                    break
                directions[layer, i] = _unit(U[:, i].to(directions.device))
            if directions[layer, 0].norm() < 1e-10:
                directions[layer, 0] = fallback
        except Exception as exc:
            # Broadcasting one vector into all k rows would ablate the same
            # direction k times and report it as a rank-k attack, so say so.
            print(
                f"  warning: SVD failed at layer {layer} ({exc}); using the "
                f"rank-1 mean difference there and leaving ranks 2..{k} empty"
            )
            directions[layer, 0] = fallback

    return directions


# --------------------------------------------------------------------------
# Refusal scoring
# --------------------------------------------------------------------------

def refusal_score(
    logits: Tensor,
    refusal_toks: Sequence[int],
    epsilon: float = 1e-8,
) -> Tensor:
    """Log-odds that the next token is a refusal token.

    Positive means the model is more likely than not to open with a refusal.
    Computed at the last position only, in float64 for stability.
    """
    logits = logits.to(torch.float64)
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, list(refusal_toks)].sum(dim=-1)
    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)


def get_refusal_scores(
    model,
    instructions: Sequence[str],
    tokenize_fn,
    refusal_toks: Sequence[int],
    *,
    fwd_pre_hooks: Sequence = (),
    fwd_hooks: Sequence = (),
    batch_size: int = 32,
) -> Tensor:
    """Refusal log-odds for each instruction, optionally under hooks."""
    score_fn = functools.partial(refusal_score, refusal_toks=refusal_toks)
    scores = torch.zeros(len(instructions), device=model.device)

    for i in range(0, len(instructions), batch_size):
        enc = tokenize_fn(instructions=list(instructions[i:i + batch_size]))
        with torch.no_grad():
            with add_hooks(
                module_forward_pre_hooks=list(fwd_pre_hooks),
                module_forward_hooks=list(fwd_hooks),
            ):
                logits = model(
                    input_ids=enc.input_ids.to(model.device),
                    attention_mask=enc.attention_mask.to(model.device),
                ).logits
        scores[i:i + batch_size] = score_fn(logits=logits)

    return scores


def filter_probes_by_refusal(
    adapter,
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    *,
    batch_size: int = 32,
) -> tuple:
    """Keep only prompts the model actually treats as harmful or harmless.

    A harmful prompt the model already complies with, or a harmless prompt it
    already refuses, adds noise to the mean difference.  Filtering on the sign of
    the refusal log-odds keeps the mean difference over prompts the model itself
    labels the same way.

    Returns ``(kept_harmful, kept_harmless)``.  A side that filters to nothing
    falls back to its full unfiltered list, because an empty side makes the mean
    difference meaningless; that is silent by design, so do not read a returned
    list as evidence that filtering kept anything.
    """
    h_scores = get_refusal_scores(
        adapter.model, harmful_instructions, adapter.tokenize_instructions_fn,
        adapter.refusal_toks, batch_size=batch_size,
    )
    s_scores = get_refusal_scores(
        adapter.model, harmless_instructions, adapter.tokenize_instructions_fn,
        adapter.refusal_toks, batch_size=batch_size,
    )
    kept_harmful = [p for p, sc in zip(harmful_instructions, h_scores.tolist()) if sc > 0]
    kept_harmless = [p for p, sc in zip(harmless_instructions, s_scores.tolist()) if sc < 0]
    return kept_harmful or list(harmful_instructions), kept_harmless or list(harmless_instructions)
