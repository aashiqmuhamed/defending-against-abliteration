"""DDO: a weight-only defense against refusal feature ablation.

Abliteration removes a model's refusal behaviour by estimating a single refusal
direction and projecting it out of the forward pass. It needs no training and no
gradient access, which is what makes it cheap to run against open weights.

DDO defends by repurposing a handful of low-impact MLP neurons into gated decoy
units. Each fires on the refusal direction and writes a learned vector
*orthogonal* to it, so an attacker re-estimating the direction on the defended
model measures something that does not remove refusal. The edit is permanent
weights, so inference cost is unchanged.

Typical use, in order:

1. :func:`ddo_defense.fingerprint.fingerprint` and ``suggest_config`` to measure
   where refusal lives in your model and get a starting layer band.
2. :func:`ddo_defense.tuning.tune` to search for a configuration. This step is
   not optional: configurations do not transfer between models, and a transferred
   one can leave a model worse than undefended.
3. :mod:`ddo_defense.attacks` to attack the result, and :mod:`ddo_eval` to score
   it with the full three-judge protocol.

Always run :func:`ddo_defense.coherence.coherence_gate` before believing any
number. A broken model refuses nothing and so scores a perfect defense.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ddo_defense.coherence import CoherenceReport, coherence_gate
from ddo_defense.data import (
    load_dataset,
    load_dataset_split,
    load_eval_set,
    load_harmbench_standard,
)
from ddo_defense.directions import (
    compute_rank_k_directions,
    estimate_refusal_directions,
    get_refusal_scores,
)
from ddo_defense.fingerprint import causal_zone, fingerprint, suggest_config
from ddo_defense.judges import LlamaGuard2Judge, SubstringJudge
from ddo_defense.mlp import get_activation_fn, get_glu_handles
from ddo_defense.models import ModelAdapter

__all__ = [
    "__version__",
    # Models and data
    "ModelAdapter",
    "load_dataset",
    "load_dataset_split",
    "load_eval_set",
    "load_harmbench_standard",
    # Directions
    "estimate_refusal_directions",
    "compute_rank_k_directions",
    "get_refusal_scores",
    # Measurement
    "fingerprint",
    "causal_zone",
    "suggest_config",
    "coherence_gate",
    "CoherenceReport",
    # MLP access
    "get_glu_handles",
    "get_activation_fn",
    # Judges
    "SubstringJudge",
    "LlamaGuard2Judge",
]


def __getattr__(name: str):
    """Expose the heavier entry points without importing them at package import."""
    if name == "tune":
        from ddo_defense.tuning import tune

        return tune
    if name == "apply_ddo_to_model":
        from ddo_defense.defense.surgery import apply_ddo_to_model

        return apply_ddo_to_model
    if name == "DDOOptimizer":
        from ddo_defense.defense.optimizer import DDOOptimizer

        return DDOOptimizer
    if name == "apply_debiasing":
        from ddo_defense.defense.debias import apply_debiasing

        return apply_debiasing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
