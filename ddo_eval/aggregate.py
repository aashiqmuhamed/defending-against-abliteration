"""Combine per-judge scores into reportable rows.

Attack success is reported as the mean over three judges, because any single
judge has characteristic failure modes and disagreement between them is itself
information.  This module keeps the per-judge numbers alongside the mean, so a
reader can see when the judges disagree rather than only the averaged figure.

Results are read from a directory laid out one subdirectory per model:

.. code-block:: text

    results_dir/
      <model_id>/
        mmlu.json                              {"score": float}
        mtbench.json                           {"score": float}
        xstest.json                            {"score": float}
        attack_<Attack>.json                   {"completions": [...]}
        scores_<judge>_attack_<Attack>.json    {"asr": float, ...}
        judging_<Attack>.json                  {"asr": {judge: float},
                                                "complete": bool,
                                                "judges_requested": [...], ...}

``judging_<Attack>.json`` is the file the reported mean comes from; the
``scores_`` files are the per-judge detail beside it.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

from ddo_eval.judges import DEFAULT_JUDGES

CAPABILITY_BENCHMARKS = ("mmlu", "mtbench", "xstest")


def mean_asr(
    judge_scores: Dict[str, Optional[float]],
    required_judges: Sequence[str] = DEFAULT_JUDGES,
) -> Optional[float]:
    """Mean for a complete requested judge set; incomplete runs have no mean."""
    if not required_judges:
        return None
    vals = [judge_scores.get(name) for name in required_judges]
    if any(v is None for v in vals):
        return None
    # A judge that wrote 2.0 where it meant 0.02 is a different failure from a
    # judge that did not run, and folding them together hides it.
    bad = [
        (name, v) for name, v in zip(required_judges, vals)
        if not math.isfinite(v) or not 0 <= v <= 1
    ]
    if bad:
        raise ValueError(
            f"ASR outside [0, 1] from {', '.join(f'{n}={v}' for n, v in bad)}. "
            f"A rate reported as a percentage would be one cause."
        )
    return float(sum(vals) / len(vals))


def aggregate_results(results_dir: str) -> Dict[str, Any]:
    """Read every model directory and build the combined table.

    Returns ``{"models": {...}, "attacks": [...], "judges": [...]}`` where each
    model carries ``capability``, ``asr`` and ``evaluation`` maps.  The third
    records, per attack, which judges were requested and ran, whether the set is
    complete, any errors, the invalid-judgment counts, the completion count and
    the scoring protocols -- it is what decides whether a mean is reported at
    all.  Every attack's ``asr``
    entry holds one value per judge plus ``avg``.
    """
    table: Dict[str, Any] = {"models": {}, "attacks": [], "judges": []}
    attacks_seen, judges_seen = set(), set()

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for model_id in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_id)
        if not os.path.isdir(model_dir):
            continue

        entry: Dict[str, Any] = {"capability": {}, "asr": {}, "evaluation": {}}

        for bench in CAPABILITY_BENCHMARKS:
            path = os.path.join(model_dir, f"{bench}.json")
            if os.path.exists(path):
                with open(path) as fh:
                    payload = json.load(fh)
                entry["capability"][bench] = payload.get("score")
                if payload.get("error"):
                    entry.setdefault("capability_errors", {})[bench] = payload["error"]

        for fname in sorted(os.listdir(model_dir)):
            if not (fname.startswith("scores_") and fname.endswith(".json")):
                continue
            stem = fname[len("scores_"):-len(".json")]
            if "_attack_" not in stem:
                continue
            judge_name, attack_name = stem.split("_attack_", 1)

            attacks_seen.add(attack_name)
            judges_seen.add(judge_name)

            with open(os.path.join(model_dir, fname)) as fh:
                entry["asr"].setdefault(attack_name, {})[judge_name] = json.load(fh).get("asr")

        # A run manifest supersedes old per-judge files, including scores left
        # behind by an earlier run whose judge failed on the latest attempt.
        for fname in sorted(os.listdir(model_dir)):
            if not (fname.startswith("judging_") and fname.endswith(".json")):
                continue
            with open(os.path.join(model_dir, fname)) as fh:
                run = json.load(fh)
            attack = run.get("attack", fname[len("judging_"):-5])
            entry["asr"][attack] = dict(run["asr"])
            # Only fields the manifest actually carries.  Materialising an
            # absent key as None makes `status.get("complete", True)` read None,
            # and `avg is not None and None` is None, so a run with three valid
            # judge scores would be reported as incomplete with no average.
            entry["evaluation"][attack] = {
                key: run[key] for key in (
                    "judges_requested", "judges_run", "complete", "errors",
                    "invalid_judgments", "n_completions", "partial_mean_asr",
                    "scoring_protocols",
                ) if key in run
            }
            attacks_seen.add(attack)
            judges_seen.update(run.get("judges_requested", []))

        for attack, scores in entry["asr"].items():
            status = entry["evaluation"].setdefault(attack, {
                "judges_requested": list(DEFAULT_JUDGES),
                "judges_run": [k for k, v in scores.items() if v is not None],
            })
            required = status.get("judges_requested") or DEFAULT_JUDGES
            avg = mean_asr(scores, required)
            complete = avg is not None and status.get("complete", True)
            scores["avg"] = avg if complete else None
            status["complete"] = complete

        table["models"][model_id] = entry

    table["attacks"] = sorted(attacks_seen)
    table["judges"] = sorted(judges_seen)
    return table


def aggregate_from_scores(
    per_judge: Dict[str, Dict[str, float]],
    capability: Optional[Dict[str, float]] = None,
    *,
    required_judges: Sequence[str] = DEFAULT_JUDGES,
) -> Dict[str, Any]:
    """Build one model's row from in-memory scores.

    ``per_judge`` maps attack name to ``{judge: asr}``.  Useful when scoring
    inside a single process rather than reading result files.
    """
    asr: Dict[str, Dict[str, Optional[float]]] = {}
    for attack, scores in per_judge.items():
        row = dict(scores)
        row["avg"] = mean_asr(row, required_judges)
        asr[attack] = row
    return {"capability": dict(capability or {}), "asr": asr}


def _fmt(val: Optional[float], *, pct: bool = True) -> str:
    if val is None:
        return "-"
    return f"{val * 100:.1f}" if pct else f"{val:.3f}"


def write_csv(table: Dict[str, Any], output_path: str) -> None:
    """Write one row per model, with per-judge columns beside each mean."""
    attacks = table.get("attacks", [])
    judges = table.get("judges", [])

    header = ["Model", "MMLU", "MT-Bench", "XSTest"]
    for attack in attacks:
        header.append(f"{attack} (avg)")
        header.append(f"{attack} (complete)")
        header.append(f"{attack} (judges requested)")
        header.extend(f"{attack} ({judge})" for judge in judges)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for model_id, data in table["models"].items():
            cap = data.get("capability", {})
            row = [
                model_id,
                _fmt(cap.get("mmlu")),
                _fmt(cap.get("mtbench"), pct=False),
                _fmt(cap.get("xstest")),
            ]
            for attack in attacks:
                scores = data.get("asr", {}).get(attack, {})
                row.append(_fmt(scores.get("avg")))
                status = data.get("evaluation", {}).get(attack, {})
                row.append(status.get("complete", scores.get("avg") is not None))
                row.append(";".join(status.get("judges_requested") or DEFAULT_JUDGES))
                row.extend(_fmt(scores.get(judge)) for judge in judges)
            writer.writerow(row)


def format_table(table: Dict[str, Any]) -> str:
    """Render a readable summary, capability first then per-attack means."""
    models = table.get("models", {})
    attacks = table.get("attacks", [])
    if not models:
        return "No results found."

    lines: List[str] = []
    lines.append(f"{'Model':<30} {'MMLU':>8} {'MT-Bench':>10} {'XSTest':>8}")
    lines.append("-" * 60)
    for model_id, data in models.items():
        cap = data.get("capability", {})
        lines.append(
            f"{model_id:<30} {_fmt(cap.get('mmlu')):>8} "
            f"{_fmt(cap.get('mtbench'), pct=False):>10} {_fmt(cap.get('xstest')):>8}"
        )

    if attacks:
        width = max(15, max(len(a) for a in attacks) + 2)
        lines.append("")
        lines.append(f"{'Model':<30}" + "".join(f" {a:>{width}}" for a in attacks))
        lines.append("-" * (30 + (width + 1) * len(attacks)))
        for model_id, data in models.items():
            row = f"{model_id:<30}"
            for attack in attacks:
                row += f" {_fmt(data.get('asr', {}).get(attack, {}).get('avg')):>{width}}"
            lines.append(row)

        for model_id, data in models.items():
            for attack, status in data.get("evaluation", {}).items():
                requested = status.get("judges_requested") or DEFAULT_JUDGES
                completed = status.get("judges_run") or []
                if not status.get("complete"):
                    lines.append(
                        f"{model_id} / {attack}: incomplete "
                        f"({len(completed)}/{len(requested)} judges); no aggregate ASR"
                    )
                elif tuple(requested) != tuple(DEFAULT_JUDGES):
                    lines.append(f"{model_id} / {attack}: judges = {', '.join(requested)}")

    return "\n".join(lines)


def save(table: Dict[str, Any], results_dir: str) -> Dict[str, str]:
    """Write ``results.json`` and ``results.csv`` beside the inputs."""
    json_path = os.path.join(results_dir, "results.json")
    csv_path = os.path.join(results_dir, "results.csv")
    os.makedirs(results_dir, exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(table, fh, indent=2)
    write_csv(table, csv_path)
    return {"json": json_path, "csv": csv_path}
