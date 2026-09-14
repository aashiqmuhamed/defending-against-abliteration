"""MMLU, to check the defense did not cost general capability.

Run through lighteval, five-shot across all 57 subjects, and reported as the mean
subject accuracy.  The version is pinned: lighteval 0.13 removed the parsing
surface this depends on, so 0.6.2 is required.

Reports exact match (em), as specified in the paper. Results containing only
other metric keys are not substituted for exact match.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

#: All 57 subjects, in the task-name form lighteval 0.6.2 expects.
MMLU_SUBJECTS: List[str] = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

REQUIRED_LIGHTEVAL = "0.6.2"


def _parse_results(results_dir: str) -> Dict[str, Any]:
    """Average accuracy across subjects from lighteval's output JSON."""
    if not os.path.isdir(results_dir):
        return {"score": None, "error": "no results directory"}

    # lighteval writes a new timestamped results_<ts>.json on every run into the
    # same directory, and os.walk yields filesystem order, so taking the first
    # match could report a previous run's score.  Newest by mtime, and ties
    # broken by name so the choice is deterministic.
    candidates = [
        os.path.join(root, fname)
        for root, _dirs, files in os.walk(results_dir)
        for fname in files
        if fname.startswith("results") and fname.endswith(".json")
    ]
    candidates.sort(key=lambda p: (os.path.getmtime(p), p), reverse=True)

    for path in candidates:
        with open(path) as fh:
            data = json.load(fh)
        results = data.get("results", {})

        accs, per_subject, missing = [], {}, []
        for key, val in results.items():
            if "mmlu" not in key.lower():
                continue
            acc = val.get("em")
            if acc is None:
                missing.append(key)
            else:
                accs.append(float(acc))
                per_subject[key] = float(acc)

        if missing:
            return {
                "score": None, "metric": "em", "missing_subjects": missing,
                "error": "Exact-match (em) scores are missing; no alternate metric was used",
                "results_file": path,
            }
        if accs:
            return {
                "score": sum(accs) / len(accs),
                "metric": "em",
                "n_subjects": len(accs),
                "n_subjects_expected": len(MMLU_SUBJECTS),
                "per_subject": per_subject,
                "results_file": path,
            }

    return {"score": None, "error": "could not parse lighteval output"}


def run_mmlu(
    model_path: str,
    output_dir: str,
    *,
    dtype: str = "bfloat16",
    max_samples: Optional[int] = 100,
    batch_size: int = 8,
    num_gpus: int = 1,
    trust_remote_code: bool = False,
    timeout: int = 7200,
) -> Dict[str, Any]:
    """Run MMLU and return ``{"score", "n_subjects", "per_subject"}``.

    ``max_samples`` caps questions per subject, which is what makes this usable
    as a regression check rather than a full evaluation. Pass ``None`` for the
    complete benchmark.
    """
    if shutil.which("lighteval") is None:
        raise RuntimeError(
            f"The 'lighteval' command was not found. Install the pinned version "
            f"with: pip install 'lighteval[accelerate]=={REQUIRED_LIGHTEVAL}'. "
            f"Later versions removed the interface this depends on."
        )

    results_dir = os.path.join(output_dir, "lighteval_mmlu")
    os.makedirs(results_dir, exist_ok=True)

    model_args = f"pretrained={model_path},dtype={dtype}"
    if trust_remote_code:
        model_args += ",trust_remote_code=True"

    tasks = ",".join(f"helm|mmlu:{subject}|5|1" for subject in MMLU_SUBJECTS)

    cmd: List[str] = []
    if num_gpus > 1:
        cmd += ["accelerate", "launch", "--multi_gpu", "--num_processes", str(num_gpus)]
        cmd += [shutil.which("lighteval"), "accelerate"]
    else:
        cmd += ["lighteval", "accelerate"]

    cmd += [
        "--model_args", model_args,
        "--tasks", tasks,
        "--override_batch_size", str(batch_size),
        "--output_dir", results_dir,
    ]
    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]

    def _write(result):
        # Written on every path, including failure: a crashed or timed-out run
        # used to leave no mmlu.json, which the aggregator cannot tell apart
        # from a benchmark that was never run.
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "mmlu.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        return result

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return _write({"score": None, "error": proc.stderr[-800:]})
    except subprocess.TimeoutExpired:
        return _write({"score": None, "error": f"lighteval timed out after {timeout}s"})

    result = _parse_results(results_dir)
    result["max_samples"] = max_samples
    return _write(result)
