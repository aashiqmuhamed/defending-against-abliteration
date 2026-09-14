"""DDO weight surgery: repurpose MLP neurons into gated decoy units.

Each hijacked neuron ``j`` is rewritten so that it computes

    delta_j(h) = act(beta * <a_i, h>) * <a_i, h> * u_j

where ``a_i`` is the direction the neuron reads, written into its gate and up
rows, and ``u_j`` is the decoy direction written into the matching ``down_proj``
column.  ``u_j`` is orthogonal to the refusal direction.  ``act`` is the layer's
own activation, SiLU for SwiGLU models and GELU-tanh for GeGLU models such as
Gemma-2.

Two compile modes are supported:

``replace``
    Overwrite the selected rows and columns.  The neuron's original function is
    discarded, so pick low-norm neurons to limit collateral damage.
``additive``
    Leave the original rows in place and superpose the trigger and decoy on them.
    This does *not* preserve the neuron's original contribution: the three rows
    multiply, so the neuron computes
    ``act(<gate + beta*a, h>) * <up + a, h> * (down + s*u)``, which is neither the
    original nor the decoy but a cross-product of both.  Gentler than discarding
    the rows outright, and the mode to try when ``replace`` fails the coherence
    gate.

Fused ``gate_up_proj`` layouts (GLM-4) are handled through
:mod:`ddo_defense.mlp`, so no model-specific branching is needed here.

Threat model: rank-k mean-difference RFA plus prompt-level jailbreaks.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ddo_defense.defense.basis import build_decoy_basis
from ddo_defense.mlp import get_glu_handles


def _normalize(v: Tensor, eps: float = 1e-8) -> Tensor:
    return v / (v.norm() + eps)


# ---------------------------------------------------------------------------
# Reader construction
# ---------------------------------------------------------------------------

def build_trigger_set(
    r: Tensor,
    n_triggers: int = 1,
    trigger_source: str = "single",
    seed: int = 0,
    gamma: float = 0.3,
) -> Tensor:
    """Build the directions the hijacked neurons read.

    Returns ``[d_model, n]`` with unit-norm columns, where ``n`` is
    ``n_triggers`` for ``"diversified"`` and always 1 for ``"single"`` -- one
    shared reader is one column however many groups are requested.  The columns are
    deliberately not orthogonal: every reader has to keep a large component along
    the refusal direction, because a reader orthogonal to it would carry no
    refusal signal and the neuron would gate on noise.

    Parameters
    ----------
    r : Tensor [d_model]
        Refusal direction for one layer.
    n_triggers : int
        Number of reader directions to build.
    trigger_source : str
        ``"single"`` for one shared reader, or ``"diversified"`` for one
        perturbed reader per decoy group.
    seed : int
        Seed for the perturbations, so a layer's readers are reproducible.
    gamma : float
        Reader diversity for ``"diversified"``: group ``g`` reads
        ``normalize(r_hat + gamma * z_g)`` with ``z_g`` random and orthogonal to
        ``r_hat``.  Ignored by ``"single"``.
    """
    device = r.device
    dtype = r.dtype
    d_model = r.shape[0]
    r_hat = _normalize(r)

    if trigger_source == "single":
        return r_hat.unsqueeze(1)  # [d_model, 1]

    if trigger_source == "diversified":
        # Perturbed readers: q_g = normalize(r_hat + gamma * z_g), z_g orth r_hat.
        # Each group still reads mostly the refusal coordinate, so it still fires
        # on harmful prompts, but the coordinate differs per group, which is what
        # gives the decoy response matrix more than one strong mode.  Column 0 is
        # r_hat itself (z = 0), so n_triggers=1 reduces exactly to the shared
        # reader and rank-1 strength is unchanged.
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        cols = [r_hat]
        if n_triggers > 1:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            Z = torch.randn(
                d_model, n_triggers - 1, generator=g,
                device=device, dtype=torch.float32,
            )
            r_hat_f = r_hat.float()
            Z = Z - torch.outer(r_hat_f, r_hat_f) @ Z
            Z = Z / (Z.norm(dim=0, keepdim=True) + 1e-8)
            for j in range(n_triggers - 1):
                q = r_hat_f + gamma * Z[:, j]
                cols.append((q / (q.norm() + 1e-8)).to(dtype=dtype))
        return torch.stack(cols, dim=1)

    raise ValueError(
        f"Unknown trigger_source '{trigger_source}'. "
        "Choose from: single, diversified"
    )


# ---------------------------------------------------------------------------
# Neuron selection
# ---------------------------------------------------------------------------

def select_neuron_indices(
    n_neurons: int,
    intermediate_size: int,
    method: str = "last",
    down_proj_weight: Optional[Tensor] = None,
    seed: int = 0,
) -> List[int]:
    """Select which MLP neurons to overwrite.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to select.
    intermediate_size : int
        Total intermediate dimension of the MLP.
    method : str
        ``"last"`` | ``"random"`` | ``"low_norm"``
    down_proj_weight : Tensor [d_model, intermediate], optional
        Required for ``"low_norm"`` selection.
    seed : int
        Random seed for ``"random"`` selection.
    """
    if method == "last":
        return list(range(intermediate_size - n_neurons, intermediate_size))

    elif method == "random":
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(intermediate_size, generator=g)
        return sorted(perm[:n_neurons].tolist())

    elif method == "low_norm":
        if down_proj_weight is None:
            raise ValueError(
                "neuron_selection='low_norm' requires down_proj_weight"
            )
        # Column norms of down_proj: [intermediate]
        col_norms = down_proj_weight.float().norm(dim=0)
        _, indices = col_norms.topk(n_neurons, largest=False)
        return sorted(indices.tolist())

    else:
        raise ValueError(
            f"Unknown neuron_selection method '{method}'. "
            "Choose from: last, random, low_norm"
        )


# ---------------------------------------------------------------------------
# Per-layer defense application
# ---------------------------------------------------------------------------

def apply_ddo_to_layer(
    layer_module,
    *,
    r: Tensor,
    n_decoys: int = 4,
    beta: float = 5.0,
    neuron_indices: Optional[Sequence[int]] = None,
    decoy_scale: float = 1.0,
    decoy_dirs_override: Optional[Tensor] = None,
    decoy_orth_to: Optional[Tensor] = None,
    seed: int = 0,
    trigger_seed: Optional[int] = None,
    n_triggers: int = 1,
    trigger_source: str = "single",
    reader_gamma: float = 0.3,
    neuron_selection: str = "last",
    compile_mode: str = "replace",
) -> Tuple[Sequence[int], Dict[str, Any]]:
    """Hijack ``n_decoys`` intermediate neurons in one layer.

    Each hijacked neuron computes ``act(beta * <a_i, h>) * <a_i, h> * u_j``,
    where ``a_i`` is its reader and ``u_j`` a decoy direction orthogonal to the
    refusal direction.

    Parameters
    ----------
    r : Tensor [d_model]
        The direction the neurons read.  Normally the refusal direction; with
        diversified readers, the group's reader.
    n_decoys : int
        Neurons to hijack.
    beta : float
        Gate sensitivity.
    neuron_indices : Sequence[int], optional
        Specific neuron indices to modify.  Selected by ``neuron_selection``
        when omitted.
    decoy_scale : float
        Magnitude of the decoy output vectors.
    decoy_dirs_override : Tensor [d_model, n_decoys], optional
        Directions to write instead of freshly built ones, which is how an
        optimised decoy is compiled.
    decoy_orth_to : Tensor [d_model], optional
        Direction the decoy columns are kept orthogonal to.  Defaults to ``r``.
        Pass the refusal direction when ``r`` is a diversified reader, so a decoy
        optimised orthogonal to refusal is not rotated at compile time.
    seed : int
        Seed for basis construction and neuron selection.
    trigger_seed : int, optional
        Separate seed for reader construction, so readers can be shared across
        layers while the basis and neuron choice vary.  Defaults to ``seed``.
    n_triggers, trigger_source, reader_gamma
        Reader configuration, passed to :func:`build_trigger_set`.
    neuron_selection : str
        ``"last"`` | ``"random"`` | ``"low_norm"``.
    compile_mode : str
        ``"replace"`` or ``"additive"``.

    Returns
    -------
    (neuron_indices, info_dict)
    """
    if trigger_seed is None:
        trigger_seed = seed

    up_proj, gate_proj, down_proj = get_glu_handles(layer_module)
    device = up_proj.weight.device
    dtype = up_proj.weight.dtype

    r = r.to(device=device, dtype=dtype)
    if r.float().norm() < 1e-8:
        raise ValueError(
            "The refusal direction for this layer is degenerate (norm ~0). "
            "Normalising it yields zeros, so the trigger written into the gate "
            "and up rows would be all-zero: the hijacked neuron would go dead "
            "and contribute nothing, giving no defense while still damaging the "
            "model. Exclude such layers from the target band. Layer 0 is always "
            "degenerate, because its last-token activation is the embedding of "
            "the template suffix that every prompt shares."
        )
    r_hat = _normalize(r)
    # Decoys are orthogonal to the refusal direction, which is not always the
    # direction the neuron reads: with diversified readers ``r`` is the group's
    # reader while the decoys still have to stay orthogonal to refusal itself.
    orth_ref = r_hat if decoy_orth_to is None else _normalize(
        decoy_orth_to.to(device=r_hat.device, dtype=r_hat.dtype)
    )

    d_model = r_hat.shape[0]
    intermediate = up_proj.weight.shape[0]

    if compile_mode not in ("replace", "additive"):
        raise ValueError(
            f"Unknown compile_mode {compile_mode!r}. Choose from: replace, additive"
        )

    if neuron_indices is not None and len(neuron_indices) != n_decoys:
        raise ValueError(
            f"n_decoys={n_decoys} but {len(neuron_indices)} neuron_indices were "
            f"given. One decoy direction is built per neuron, so the two must "
            f"agree; otherwise the layer is left half-modified."
        )

    if neuron_indices is None:
        neuron_indices = select_neuron_indices(
            n_neurons=n_decoys,
            intermediate_size=intermediate,
            method=neuron_selection,
            down_proj_weight=down_proj.weight.data if neuron_selection == "low_norm" else None,
            seed=seed,
        )

    triggers = build_trigger_set(
        r_hat, n_triggers=n_triggers, trigger_source=trigger_source,
        seed=trigger_seed, gamma=reader_gamma,
    ).to(device=device, dtype=dtype)  # [d_model, n_triggers]

    if decoy_dirs_override is not None:
        D = decoy_dirs_override.to(device=device, dtype=torch.float32)
        if D.shape != (d_model, n_decoys):
            raise ValueError(
                "decoy_dirs_override must have shape "
                f"({d_model}, {n_decoys}), got {tuple(D.shape)}"
            )
        # Enforce orthogonality to the refusal direction and re-orthonormalize.
        D = D - torch.outer(orth_ref.to(D), orth_ref.to(D)) @ D
        Q, _ = torch.linalg.qr(D)
        # QR is unique up to per-column sign. Preserve sign orientation relative
        # to the provided columns so optimised directions compile faithfully.
        dots = (Q * D).sum(dim=0)
        signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
        Q = Q * signs
        decoy_dirs = Q[:, :n_decoys].to(device=device, dtype=dtype)
    else:
        # Built around the direction the decoys must avoid, which is not always
        # the one the neuron reads: with a diversified reader, r is the group's
        # reader while the decoys still have to stay orthogonal to refusal.
        U = build_decoy_basis(orth_ref, k=n_decoys + 1, seed=seed, dtype=torch.float32)
        decoy_dirs = U[:, 1:].to(device=device, dtype=dtype)

    with torch.no_grad():
        for j, idx in enumerate(neuron_indices):
            # Readers cycle when there are more neurons than reader groups.
            trigger = triggers[:, j % triggers.shape[1]]

            # up_proj row reads <a_i, h>; gate_proj row reads beta * <a_i, h>
            # so that the activation sees the scaled projection.
            new_up = trigger
            new_gate = beta * trigger
            new_down = decoy_scale * decoy_dirs[:, j]

            if compile_mode == "replace":
                up_proj.weight[idx].copy_(new_up)
                gate_proj.weight[idx].copy_(new_gate)
                down_proj.weight[:, idx].copy_(new_down)
            else:  # additive
                up_proj.weight[idx].add_(new_up)
                gate_proj.weight[idx].add_(new_gate)
                down_proj.weight[:, idx].add_(new_down)

    info = {
        "n_triggers": n_triggers,
        "trigger_source": trigger_source,
        "n_decoy_neurons": n_decoys,
        "neuron_selection": neuron_selection,
        "compile_mode": compile_mode,
    }

    return neuron_indices, info


# ---------------------------------------------------------------------------
# Model-level defense application
# ---------------------------------------------------------------------------

def apply_ddo_to_model(
    model,
    *,
    r_by_layer: Sequence[Tensor],
    target_layers: Optional[Iterable[int]] = None,
    n_decoys: int = 4,
    beta: float = 5.0,
    decoy_scale: float = 1.0,
    seed: int = 0,
    n_triggers: int = 1,
    trigger_source: str = "single",
    reader_gamma: float = 0.3,
    vary_triggers_across_layers: bool = False,
    neuron_selection: str = "last",
    compile_mode: str = "replace",
) -> Dict[str, object]:
    """Apply DDO injection to several layers of a Llama-like model.

    Parameters
    ----------
    r_by_layer : Sequence[Tensor]
        Per-layer refusal directions, one ``[d_model]`` vector per layer.
    target_layers : Iterable[int], optional
        Layer indices to defend.  Defaults to every layer.
    n_decoys : int
        Neurons to hijack per layer.
    beta, decoy_scale
        Gate sensitivity and decoy magnitude.
    seed : int
        Base seed; the per-layer basis seed is ``seed + layer``.
    n_triggers, trigger_source, reader_gamma
        Reader configuration, passed through to each layer.
    vary_triggers_across_layers : bool
        Use ``seed + layer`` rather than ``seed`` for reader construction.
    neuron_selection : str
        ``"last"`` | ``"random"`` | ``"low_norm"``.
    compile_mode : str
        ``"replace"`` or ``"additive"``.

    Returns
    -------
    Dict with the defense configuration and what was written.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected a Llama-like model with model.model.layers")

    n_layers = len(layers)
    if len(r_by_layer) != n_layers:
        raise ValueError(
            f"r_by_layer must be one vector per layer, got "
            f"{len(r_by_layer)} for {n_layers} layers"
        )

    if target_layers is None:
        target_layers = list(range(n_layers))
    else:
        target_layers = list(target_layers)

    # Drop layers with no refusal signal rather than lobotomising a neuron there.
    skipped_layers = [
        l for l in target_layers if r_by_layer[l].float().norm() < 1e-8
    ]
    if skipped_layers:
        print(
            f"  skipping {len(skipped_layers)} layer(s) with a degenerate "
            f"refusal direction: {skipped_layers}"
        )
        target_layers = [l for l in target_layers if l not in skipped_layers]
    if not target_layers:
        raise ValueError(
            "Every requested layer has a degenerate refusal direction, so there "
            "is nothing to build a decoy around. Estimate directions from more "
            "prompts, or choose a deeper band."
        )

    print(f"Applying DDO injection to {len(target_layers)} layers...")
    print(f"  n_decoys = {n_decoys} neurons per layer")
    print(f"  beta = {beta} (gate sensitivity)")
    print(f"  decoy_scale = {decoy_scale}")
    print(f"  n_triggers = {n_triggers} ({trigger_source})")
    print(f"  neuron_selection = {neuron_selection}")
    print(f"  compile_mode = {compile_mode}")

    all_neuron_indices: Dict[int, Sequence[int]] = {}
    all_layer_info: Dict[int, object] = {}

    for layer_idx in target_layers:
        # Readers are shared across layers unless asked to vary; the basis and
        # neuron choice always vary, so layers do not all get the same decoy.
        trigger_seed = (seed + layer_idx) if vary_triggers_across_layers else seed
        basis_seed = seed + layer_idx

        indices, info = apply_ddo_to_layer(
            layers[layer_idx],
            r=r_by_layer[layer_idx],
            n_decoys=n_decoys,
            beta=beta,
            decoy_scale=decoy_scale,
            seed=basis_seed,
            trigger_seed=trigger_seed,
            n_triggers=n_triggers,
            trigger_source=trigger_source,
            reader_gamma=reader_gamma,
            neuron_selection=neuron_selection,
            compile_mode=compile_mode,
        )
        all_neuron_indices[layer_idx] = indices
        all_layer_info[layer_idx] = info

    print("Gated decoy injection applied.")

    return {
        "defense_type": "ddo",
        "compile_mode": compile_mode,
        "target_layers": target_layers,
        "skipped_layers": skipped_layers,
        "n_decoys": n_decoys,
        "beta": beta,
        "decoy_scale": decoy_scale,
        "n_triggers": n_triggers,
        "trigger_source": trigger_source,
        "reader_gamma": reader_gamma,
        "neuron_selection": neuron_selection,
        "vary_triggers_across_layers": vary_triggers_across_layers,
        "neuron_indices": {str(k): v for k, v in all_neuron_indices.items()},
        "layer_info": {str(k): v for k, v in all_layer_info.items()},
    }
