"""The undo attack: locate DDO's injected neurons and switch them off.

DDO writes a trigger into the gate and up rows of a few MLP neurons and a decoy
vector into the matching ``down_proj`` columns.  An attacker who suspects this
does not need the base model: the injected neurons are structurally unusual, so
a cheap per-neuron score can rank them and the top few can be ablated by zeroing
their ``down_proj`` column, which removes their contribution entirely.

Three scoring heuristics are provided, each exploiting a different signature:

``gate_proj``
    ``|W_gate[i] . r|`` -- injected gate rows are written as ``beta * r``, so
    they align with the refusal direction far more strongly than normal rows.
``gate_up_cos``
    ``|cos(W_gate[i], W_up[i])|`` -- injection writes the *same* trigger into
    both rows up to the factor ``beta``, making them nearly collinear, which is
    not true of ordinary neurons.
``down_norm``
    ``||W_down[:, i]||`` -- flags neurons whose output vector is unusually
    large, which catches injection at high ``decoy_scale``.

Report the attack at several budgets; a defense that only survives ``m=1`` is
not robust.  This is an evaluation tool for measuring your own defense.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ddo_defense.mlp import get_glu_handles

HEURISTICS = ("gate_proj", "gate_up_cos", "down_norm")


def score_neurons(layer_module, r: Optional[Tensor] = None, *, heuristic: str = "gate_proj") -> Tensor:
    """Score every neuron in one layer; higher means more likely injected.

    Parameters
    ----------
    layer_module
        A decoder layer.  Fused ``gate_up_proj`` layouts are handled.
    r
        Refusal direction for this layer, required by ``gate_proj``.  This is
        the *attacker's* estimate, recomputed on the defended model.
    heuristic
        One of :data:`HEURISTICS`.

    Returns
    -------
    Tensor
        One score per neuron, shape ``[intermediate_size]``.
    """
    if heuristic not in HEURISTICS:
        raise ValueError(f"Unknown heuristic {heuristic!r}; choose from {HEURISTICS}")

    up, gate, down = get_glu_handles(layer_module)

    if heuristic == "gate_proj":
        if r is None:
            raise ValueError("heuristic='gate_proj' requires the refusal direction r")
        gate_w = gate.weight.data.float()
        r_vec = r.detach().float().to(gate_w.device)
        if r_vec.norm() < 1e-8:
            raise ValueError(
                "heuristic='gate_proj' needs a refusal direction, and this one is "
                "degenerate (norm ~0). Every neuron would score 0, so the attack "
                "would zero whichever m neurons the tie-break happened to pick. "
                "Layer 0 is always degenerate; exclude such layers."
            )
        r_vec = r_vec / r_vec.norm()
        return (gate_w @ r_vec).abs()

    if heuristic == "gate_up_cos":
        gate_w = gate.weight.data.float()
        up_w = up.weight.data.float()
        return F.cosine_similarity(gate_w, up_w, dim=1).abs()

    return down.weight.data.float().norm(dim=0)


def undo_layer(
    layer_module,
    r: Optional[Tensor] = None,
    *,
    m: int = 8,
    heuristic: str = "gate_proj",
) -> List[int]:
    """Zero the ``down_proj`` columns of the top-``m`` neurons in one layer.

    Returns the neuron indices that were ablated.  Mutates the layer in place.
    """
    scores = score_neurons(layer_module, r, heuristic=heuristic)
    m_eff = int(min(m, scores.shape[0]))
    if m_eff <= 0:
        return []

    _, top_idx = scores.topk(m_eff)
    _up, _gate, down = get_glu_handles(layer_module)
    with torch.no_grad():
        for idx in top_idx.tolist():
            down.weight.data[:, idx] = 0
    return sorted(int(i) for i in top_idx.tolist())


def undo_attack(
    model,
    *,
    r_by_layer: Optional[Sequence[Tensor]] = None,
    target_layers: Optional[Iterable[int]] = None,
    m: int = 8,
    heuristic: str = "gate_proj",
) -> Dict[str, object]:
    """Run the undo attack across layers, mutating ``model`` in place.

    Parameters
    ----------
    model
        A loaded causal LM exposing ``model.model.layers``.
    r_by_layer
        Per-layer refusal directions, required for the ``gate_proj`` heuristic.
        These should be re-estimated on the *defended* model, since a real
        attacker has no access to the undefended one.
    target_layers
        Layers to attack.  Defaults to every layer.  When the defended band is
        known, restricting to it makes the attack strictly stronger.
    m
        Neurons ablated per layer.
    heuristic
        One of :data:`HEURISTICS`.

    Returns
    -------
    dict
        ``{"heuristic", "m", "ablated": {layer: [indices]}}``.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected a model exposing model.model.layers")

    if target_layers is None:
        target_layers = range(len(layers))
    target_layers = list(target_layers)

    if heuristic == "gate_proj" and r_by_layer is None:
        raise ValueError("heuristic='gate_proj' requires r_by_layer")

    ablated: Dict[int, List[int]] = {}
    for li in target_layers:
        r = r_by_layer[li] if r_by_layer is not None else None
        ablated[li] = undo_layer(layers[li], r, m=m, heuristic=heuristic)

    return {"heuristic": heuristic, "m": m, "ablated": ablated}
