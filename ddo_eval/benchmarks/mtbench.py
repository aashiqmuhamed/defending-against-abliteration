"""MT-Bench, the other half of the capability check.

Two stages through FastChat: generate answers to the 80 multi-turn questions,
then have GPT-4 score them.  The score is the mean over all judged turns, on a
one-to-ten scale, so it is not a percentage and is formatted differently from
every other number in the report.

Needs ``OPENAI_API_KEY`` for judging.  FastChat does not bundle the question
file, the judge prompts, or the reference answers, so a first run must download
them; :func:`check_data_files` says what is missing and where it goes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

N_QUESTIONS = 80

_DATA_FILES = {
    "data/mt_bench/question.jsonl":
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
    "data/judge_prompts.jsonl":
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/judge_prompts.jsonl",
    "data/mt_bench/reference_answer/gpt-4.jsonl":
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl",
}


def _llm_judge_dir() -> str:
    try:
        import fastchat
    except ImportError as exc:
        raise ImportError(
            "MT-Bench needs FastChat. Install the [eval] extra."
        ) from exc
    return os.path.join(os.path.dirname(fastchat.__file__), "llm_judge")


def check_data_files() -> List[str]:
    """Return the data files FastChat is missing, as absolute paths."""
    base = _llm_judge_dir()
    return [
        os.path.join(base, rel)
        for rel in _DATA_FILES
        if not os.path.exists(os.path.join(base, rel))
    ]


def fetch_data_files() -> List[str]:
    """Download the three files FastChat omits from its package."""
    import urllib.request

    base = _llm_judge_dir()
    fetched = []
    for rel, url in _DATA_FILES.items():
        dest = os.path.join(base, rel)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as resp:
            with open(dest, "wb") as fh:
                fh.write(resp.read())
        fetched.append(dest)
    return fetched


def _parse_judgments(model_id: str, expected_turns: Optional[int] = None) -> Dict[str, Any]:
    """Mean judged score for ``model_id``.

    No per-category breakdown: a judgment record carries question_id, model,
    judge, score and turn, but not category, so keying a breakdown off
    ``entry["category"]`` put every score in one "unknown" bucket.
    """
    base = _llm_judge_dir()
    path = os.path.join(base, "data", "mt_bench", "model_judgment", "gpt-4_single.jsonl")
    if not os.path.exists(path):
        return {"score": None, "error": f"no judgment file at {path}"}

    scores: List[float] = []
    with open(path) as fh:
        for line in fh:
            entry = json.loads(line)
            if entry.get("model") != model_id:
                continue
            score = entry.get("score", -1)
            if score and score > 0:
                scores.append(float(score))

    if not scores:
        return {"score": None, "error": f"no scores found for {model_id}"}

    out = {
        "score": sum(scores) / len(scores),
        "n_scored": len(scores),
    }
    if expected_turns is not None:
        out["n_turns_expected"] = expected_turns
        if len(scores) != expected_turns:
            # gpt-4_single.jsonl is shared and append-only, so a partial run or a
            # re-run can leave a different number of judged turns than asked for.
            out["error"] = (
                f"judged {len(scores)} turns, expected {expected_turns}; the "
                f"mean is over whatever was in the judgment file"
            )
    return out


def _write(output_dir: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Write the result, including on the failure paths.

    A run that failed used to leave no mtbench.json at all, which the aggregator
    cannot tell apart from a benchmark that was never run.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "mtbench.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def run_mtbench(
    model_path: str,
    output_dir: str,
    model_id: str,
    *,
    dtype: str = "bfloat16",
    num_gpus_total: int = 1,
    num_gpus_per_model: int = 1,
    judge_parallel: int = 8,
    n_questions: Optional[int] = None,
    auto_fetch_data: bool = True,
    timeout: int = 7200,
) -> Dict[str, Any]:
    """Generate answers, judge them with GPT-4, and return the mean score.

    Answers are written incrementally by FastChat, so an interrupted run resumes
    rather than starting over.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("MT-Bench judging needs OPENAI_API_KEY.")
    if num_gpus_total < num_gpus_per_model:
        raise ValueError(
            f"num_gpus_total={num_gpus_total} is fewer than "
            f"num_gpus_per_model={num_gpus_per_model}, so not one replica fits."
        )

    missing = check_data_files()
    if missing:
        if auto_fetch_data:
            fetch_data_files()
        else:
            raise RuntimeError(
                "FastChat is missing MT-Bench data files: "
                + ", ".join(missing)
                + ". Call fetch_data_files() or download them manually."
            )

    cwd = _llm_judge_dir()
    target = n_questions or N_QUESTIONS
    answer_file = os.path.join(cwd, "data", "mt_bench", "model_answer", f"{model_id}.jsonl")

    have = 0
    if os.path.exists(answer_file):
        with open(answer_file) as fh:
            have = sum(1 for _ in fh)

    if have < target:
        # Ray shards questions across replicas, so never ask for more replicas
        # than questions or the shard size rounds to zero.  FastChat derives its
        # replica count as total // per_model and asserts the division is exact,
        # so the clamp has to move in whole replicas.
        replicas = max(1, min(num_gpus_total // num_gpus_per_model, target))
        effective_gpus = replicas * num_gpus_per_model
        cmd = [
            sys.executable, "-m", "fastchat.llm_judge.gen_model_answer",
            "--model-path", model_path,
            "--model-id", model_id,
            "--num-gpus-per-model", str(num_gpus_per_model),
            "--num-gpus-total", str(effective_gpus),
            "--dtype", dtype,
        ]
        if n_questions:
            cmd += ["--question-end", str(n_questions)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
            if proc.returncode != 0:
                return _write(output_dir, {"score": None, "error": proc.stderr[-800:]})
        except subprocess.TimeoutExpired:
            return _write(
                output_dir, {"score": None, "error": "answer generation timed out"}
            )

    judge_cmd = [
        sys.executable, "-m", "fastchat.llm_judge.gen_judgment",
        "--model-list", model_id,
        "--mode", "single",
        "--judge-model", "gpt-4",
        "--parallel", str(judge_parallel),
    ]
    if n_questions:
        judge_cmd += ["--first-n", str(n_questions)]

    try:
        # gen_judgment waits on an interactive confirmation; feed it a newline.
        proc = subprocess.run(
            judge_cmd, input="\n", capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
        if proc.returncode != 0:
            return _write(output_dir, {"score": None, "error": proc.stderr[-800:]})
    except subprocess.TimeoutExpired:
        return _write(output_dir, {"score": None, "error": "judging timed out"})

    result = _parse_judgments(model_id, expected_turns=2 * target)
    return _write(output_dir, result)
