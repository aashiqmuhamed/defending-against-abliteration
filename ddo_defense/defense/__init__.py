"""The defense itself: weight surgery, its optimizer, and the over-refusal repair.

* :mod:`~ddo_defense.defense.surgery` writes decoys into MLP weights.
* :mod:`~ddo_defense.defense.optimizer` learns what to write.
* :mod:`~ddo_defense.defense.basis` builds the candidate decoy directions.
* :mod:`~ddo_defense.defense.debias` repairs over-refusal afterwards.
"""

from ddo_defense.defense.surgery import apply_ddo_to_layer, apply_ddo_to_model

__all__ = ["apply_ddo_to_layer", "apply_ddo_to_model"]
