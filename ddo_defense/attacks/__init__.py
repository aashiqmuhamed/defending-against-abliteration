"""Attacks, for measuring whether a defense actually holds.

* :mod:`~ddo_defense.attacks.rfa` is standard refusal feature ablation, with
  rank and probe budget as parameters.
* :mod:`~ddo_defense.attacks.adaptive` re-estimates the direction across phases,
  which catches a defense that moves the signal rather than removing it.
* :mod:`~ddo_defense.attacks.undo` finds and switches off the injected neurons.

Report all three. A defense evaluated only against single-phase rank-1 ablation
has not been evaluated.
"""

from ddo_defense.attacks.adaptive import n_phase_attack
from ddo_defense.attacks.rfa import RFA_VARIANTS, run_rfa, run_rfa_variant
from ddo_defense.attacks.undo import HEURISTICS, undo_attack

__all__ = [
    "run_rfa",
    "run_rfa_variant",
    "RFA_VARIANTS",
    "n_phase_attack",
    "undo_attack",
    "HEURISTICS",
]
