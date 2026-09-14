"""Over-refusal repair: recover benign compliance without weakening refusal.

Hardening a model against abliteration pushes it toward refusing more, including
prompts it should answer.  The repair estimates a direction that separates
wrongly-refused benign prompts from genuinely harmful ones, forces that direction
orthogonal to the refusal direction so the two can be moved independently, and
then uses it to pull compliance back up.

Two methods are provided behind ``method``:

``lm_head``
    Bias the output head: add to the rows of tokens that begin a compliant reply
    and subtract from the rows that begin a refusal.  A small, local edit that
    leaves the residual stream untouched.
``projection`` (default)
    Remove the direction from the matrices that write into the residual stream:
    the embedding, each attention output projection, and each MLP output
    projection.  Each can be disabled individually.

Neither method involves gradient descent.  The direction is closed-form from
forward passes, and ``boost`` and ``suppress`` are scalars; expose them to the
tuner if you want them fitted.

Whatever prompts are used to estimate the direction must be excluded from any
reported over-refusal score, or the number is measured partly on the data it was
fitted to.  The dev/report split in :mod:`ddo_eval` enforces this.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from ddo_defense.hooks import add_hooks

#: First words of a compliant reply.  Boosting these makes compliance cheaper.
DEFAULT_COMPLIANCE_WORDS: Sequence[str] = (
    "The", "Here", "In", "To", "There", "This", "A", "It", "For", "You",
)

#: First words of a refusal.  Suppressing these makes refusal costlier.
DEFAULT_REFUSAL_WORDS: Sequence[str] = ("I", "cannot")

#: Depth fractions of the four late layers the direction is read from.  On a
#: 32-layer model these are layers 24, 26, 28 and 30.
DEFAULT_LAYER_FRACTIONS: Sequence[float] = (0.75, 0.8125, 0.875, 0.9375)


def default_layers(n_layers: int) -> List[int]:
    """The late layers the over-refusal direction is read from.

    Up to four, and fewer on shallow models: the depth fractions collide there
    and the result is de-duplicated.
    """
    return sorted({min(int(n_layers * f), n_layers - 1) for f in DEFAULT_LAYER_FRACTIONS})


def _mean_last_token_acts(adapter, prompts: Sequence[str], layers: Sequence[int]) -> Dict[int, Tensor]:
    """Mean last-token block-input activation at each requested layer."""
    blocks = adapter.blocks
    sums = {l: None for l in layers}
    n = 0

    for prompt in prompts:
        storage: Dict[int, Tensor] = {}

        def make_hook(layer):
            def hook_fn(module, inp):
                act = inp[0] if isinstance(inp, tuple) else inp
                storage[layer] = act[:, -1, :].detach().float().cpu()
            return hook_fn

        hooks = [(blocks[l], make_hook(l)) for l in layers]
        enc = adapter.tokenize([prompt])
        with torch.no_grad():
            with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
                adapter.model(
                    input_ids=enc.input_ids.to(adapter.model.device),
                    attention_mask=enc.attention_mask.to(adapter.model.device),
                )
        for l in layers:
            if l in storage:
                v = storage[l].squeeze(0)
                sums[l] = v if sums[l] is None else sums[l] + v
        n += 1

    return {l: (s / max(n, 1)) for l, s in sums.items() if s is not None}


def estimate_over_refusal_direction(
    adapter,
    over_refused_prompts: Sequence[str],
    harmful_prompts: Sequence[str],
    r_by_layer: Sequence[Tensor],
    *,
    layers: Optional[Sequence[int]] = None,
) -> Tensor:
    """Unit direction separating wrongly-refused benign prompts from harmful ones.

    At each chosen layer the difference of means is taken, the refusal component
    is projected out so the repair cannot simply undo refusal, and the result is
    normalised.  The per-layer directions are then summed and normalised once
    more into a single vector.

    Parameters
    ----------
    over_refused_prompts
        Benign prompts the model wrongly refuses.  These must not appear in any
        reported over-refusal score.
    harmful_prompts
        Genuinely harmful prompts, as the contrast set.
    r_by_layer
        Per-layer refusal directions to orthogonalise against.
    layers
        Layers to read.  Defaults to :func:`default_layers`.
    """
    if not over_refused_prompts:
        raise ValueError("Need at least one over-refused prompt to estimate the direction")
    if not harmful_prompts:
        raise ValueError("Need at least one harmful prompt as the contrast set")

    layers = list(layers) if layers is not None else default_layers(adapter.n_layers)

    benign_acts = _mean_last_token_acts(adapter, list(over_refused_prompts), layers)
    harmful_acts = _mean_last_token_acts(adapter, list(harmful_prompts), layers)

    total = torch.zeros(adapter.d_model, dtype=torch.float32)
    for l in layers:
        if l not in benign_acts or l not in harmful_acts:
            continue
        diff = benign_acts[l] - harmful_acts[l]
        r = r_by_layer[l].detach().float().cpu()
        r = r / (r.norm() + 1e-8)
        orth = diff - (diff @ r) * r
        nrm = orth.norm()
        if nrm > 1e-6:
            total += orth / nrm

    # Each layer's component is orthogonal to that layer's refusal direction, but
    # their sum is not orthogonal to any one of them: the r_l differ per layer, so
    # a vector orthogonal to all of them generally does not exist.  A final pass
    # removes the component along each layer's r in turn, which leaves a residual
    # that is small against every one rather than exactly zero against one.
    for _ in range(2):
        for l in layers:
            if l not in benign_acts or l not in harmful_acts:
                continue
            r = r_by_layer[l].detach().float().cpu()
            r = r / (r.norm() + 1e-8)
            total = total - (total @ r) * r

    norm = total.norm()
    if norm < 1e-8:
        raise RuntimeError(
            "Estimated over-refusal direction is degenerate. The benign and "
            "harmful activation means are nearly identical at these layers; try "
            "different layers or a larger prompt set."
        )
    return total / norm


def _token_ids(tokenizer, words: Sequence[str]) -> List[int]:
    """First-token id of each word, de-duplicated, order preserved."""
    ids: List[int] = []
    for word in words:
        enc = tokenizer.encode(word, add_special_tokens=False)
        if enc:
            ids.append(int(enc[0]))
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def apply_lm_head_debiasing(
    adapter,
    v_hat: Tensor,
    *,
    boost: float = 0.12,
    suppress: float = 0.02,
    compliance_words: Sequence[str] = DEFAULT_COMPLIANCE_WORDS,
    refusal_words: Sequence[str] = DEFAULT_REFUSAL_WORDS,
) -> Dict[str, object]:
    """Bias the output head along ``v_hat``.

    Compliance-token rows gain ``boost * v_hat`` and refusal-token rows lose
    ``suppress * v_hat``, so a prompt whose activation lies along the direction
    finds compliance slightly cheaper and refusal slightly costlier.  The
    asymmetric defaults reflect that suppressing refusal is the riskier half.

    Token ids come from the checkpoint's own tokenizer, so this is not tied to
    any one vocabulary.
    """
    model = adapter.model
    head = getattr(model, "lm_head", None)
    if head is None:
        raise ValueError(
            "This model has no lm_head, so method='lm_head' does not apply. "
            "Use method='projection'."
        )

    weight = head.weight.data
    v = v_hat.to(device=weight.device, dtype=weight.dtype)

    tied = weights_are_tied(model)
    if tied:
        print(
            "  note: embeddings are tied to the output head, so these row edits "
            "also shift the input embedding of the same tokens. The shift is "
            "boost/suppress-sized and is recorded as tied_embeddings in the result."
        )

    comp_ids = _token_ids(adapter.tokenizer, compliance_words)
    ref_ids = _token_ids(adapter.tokenizer, refusal_words)

    # A token in both lists would be boosted and then suppressed, leaving a net
    # edit that depends only on the order of the two loops.
    both = set(comp_ids) & set(ref_ids)
    if both:
        print(
            f"  note: {len(both)} token id(s) are both compliance and refusal "
            f"prefixes in this vocabulary and are left untouched: {sorted(both)}"
        )
        comp_ids = [t for t in comp_ids if t not in both]
        ref_ids = [t for t in ref_ids if t not in both]

    vocab = weight.shape[0]
    applied_boost, applied_suppress = [], []
    with torch.no_grad():
        for tid in comp_ids:
            if 0 <= tid < vocab:
                weight[tid] += boost * v
                applied_boost.append(tid)
        for tid in ref_ids:
            if 0 <= tid < vocab:
                weight[tid] -= suppress * v
                applied_suppress.append(tid)

    return {
        "method": "lm_head",
        "tied_embeddings": tied,
        "boost": boost,
        "suppress": suppress,
        "boosted_token_ids": applied_boost,
        "suppressed_token_ids": applied_suppress,
        "compliance_words": list(compliance_words),
        "refusal_words": list(refusal_words),
    }


def _project_out_rows(weight: Tensor, v: Tensor) -> None:
    """In place: ``W <- W - (W v) v^T`` for ``W`` with ``d_model`` as columns."""
    vv = v.to(device=weight.device, dtype=torch.float32)
    W = weight.data.float()
    W -= torch.outer(W @ vv, vv)
    weight.data.copy_(W.to(weight.dtype))


def _project_out_cols(weight: Tensor, v: Tensor) -> None:
    """In place: ``W <- (I - v v^T) W`` for ``W`` with ``d_model`` as rows."""
    vv = v.to(device=weight.device, dtype=torch.float32)
    W = weight.data.float()
    W -= torch.outer(vv, vv @ W)
    weight.data.copy_(W.to(weight.dtype))


def weights_are_tied(model) -> bool:
    """True when the input embedding and the output head share one tensor.

    Gemma-2 ties by default.  It matters here because an edit to either matrix
    is then an edit to both, which changes what each method actually does.
    """
    emb = model.get_input_embeddings()
    head = getattr(model, "lm_head", None)
    if emb is None or head is None:
        return False
    if emb.weight is head.weight:
        return True
    return bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))


def apply_projection_debiasing(
    adapter,
    v_hat: Tensor,
    *,
    include_embedding: bool = True,
    include_attn_out: bool = True,
    include_mlp_out: bool = True,
) -> Dict[str, object]:
    """Remove ``v_hat`` from every weight that writes into the residual stream.

    The axis decides the form.  For ``nn.Embedding`` the residual dimension is
    the second axis, so each row is projected; for ``o_proj`` and ``down_proj`` it
    is the first, so the projection acts from the left.  Those three are every
    path that writes into the residual stream.

    On a checkpoint with tied embeddings the input embedding *is* the output
    head, so projecting the direction out of it would also strip that direction
    from every unembedding row and change every logit the model produces.  That
    is a read path, not a write path, so the embedding is left alone there and
    the result records it.
    """
    model = adapter.model
    v = v_hat.detach().float()
    v = v / (v.norm() + 1e-8)

    touched: List[str] = []
    tied = weights_are_tied(model)
    skipped_tied_embedding = False

    if include_embedding:
        emb = model.get_input_embeddings()
        if emb is None:
            pass
        elif tied:
            skipped_tied_embedding = True
            print(
                "  note: embeddings are tied to the output head, so the embedding "
                "is left unprojected; editing it would change every logit. The "
                "attention and MLP write paths are still projected."
            )
        else:
            _project_out_rows(emb.weight, v)
            touched.append("embed_tokens")

    for li, block in enumerate(adapter.blocks):
        if include_attn_out:
            attn = next(
                (getattr(block, nm) for nm in ("self_attn", "attn", "attention")
                 if getattr(block, nm, None) is not None),
                None,
            )
            o_proj = next(
                (getattr(attn, nm) for nm in ("o_proj", "dense", "c_proj")
                 if getattr(attn, nm, None) is not None),
                None,
            ) if attn is not None else None
            if o_proj is not None:
                _project_out_cols(o_proj.weight, v)
                touched.append(f"layers.{li}.self_attn.o_proj")
        if include_mlp_out:
            mlp = getattr(block, "mlp", None)
            down = next(
                (getattr(mlp, nm) for nm in ("down_proj", "dense_4h_to_h", "c_proj")
                 if getattr(mlp, nm, None) is not None),
                None,
            ) if mlp is not None else None
            if down is not None:
                _project_out_cols(down.weight, v)
                touched.append(f"layers.{li}.mlp.down_proj")

    return {
        "method": "projection",
        "n_matrices_edited": len(touched),
        "include_embedding": include_embedding,
        "include_attn_out": include_attn_out,
        "include_mlp_out": include_mlp_out,
        "tied_embeddings": tied,
        "embedding_skipped_because_tied": skipped_tied_embedding,
        "edited": touched,
    }


def apply_debiasing(
    adapter,
    over_refused_prompts: Sequence[str],
    harmful_prompts: Sequence[str],
    r_by_layer: Sequence[Tensor],
    *,
    method: str = "projection",
    layers: Optional[Sequence[int]] = None,
    boost: float = 0.12,
    suppress: float = 0.02,
    compliance_words: Sequence[str] = DEFAULT_COMPLIANCE_WORDS,
    refusal_words: Sequence[str] = DEFAULT_REFUSAL_WORDS,
    **projection_kwargs,  # only meaningful for method="projection"
) -> Dict[str, object]:
    """Estimate the direction, then apply the chosen repair.  Mutates the model.

    ``method="projection"`` is the default.  ``method="lm_head"`` is the
    lighter alternative, biasing the output head instead of editing the weights
    that write the residual stream.
    """
    if method not in ("lm_head", "projection"):
        raise ValueError(
            f"Unknown method {method!r}. Choose 'lm_head' or 'projection'."
        )
    if method == "lm_head" and projection_kwargs:
        raise ValueError(
            f"{sorted(projection_kwargs)} only apply to method='projection' and "
            f"would be silently ignored here. Drop them, or switch method."
        )

    v_hat = estimate_over_refusal_direction(
        adapter, over_refused_prompts, harmful_prompts, r_by_layer, layers=layers
    )

    if method == "lm_head":
        info = apply_lm_head_debiasing(
            adapter, v_hat, boost=boost, suppress=suppress,
            compliance_words=compliance_words, refusal_words=refusal_words,
        )
    else:
        info = apply_projection_debiasing(adapter, v_hat, **projection_kwargs)

    info.update({
        "layers_read": list(layers) if layers is not None else default_layers(adapter.n_layers),
        "n_over_refused_prompts": len(over_refused_prompts),
        "n_harmful_prompts": len(harmful_prompts),
    })
    return info
