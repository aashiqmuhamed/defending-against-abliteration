"""Evaluation: capability benchmarks, safety judges, and reporting.

Attack success is the mean over three judges, because each has its own failure
modes and their disagreement is worth seeing. :func:`ddo_eval.judges.score_all`
runs any subset and reports per-judge numbers next to the mean.

Capability coverage is MMLU, MT-Bench and XSTest. XSTest needs care: it counts a
non-refusal as compliance, so a degenerate model scores well on it. Run
:func:`ddo_defense.coherence.coherence_gate` first.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ddo_eval.aggregate import aggregate_results, format_table, mean_asr
from ddo_eval.judges import DEFAULT_JUDGES, JUDGE_NAMES, create_judge, score_all

__all__ = [
    "__version__",
    "create_judge",
    "score_all",
    "JUDGE_NAMES",
    "DEFAULT_JUDGES",
    "aggregate_results",
    "mean_asr",
    "format_table",
]
