"""Result parsing for the capability benchmarks.

Both benchmarks shell out to an external harness, so the only logic worth testing
is how their output is read back. That logic matters more than it looks: a parser
that picks the wrong metric key, counts the aggregate row as a subject, or fails
to filter judgments by model reports a plausible-looking number that is simply
wrong, and it would land in a results table unchallenged.
"""

from __future__ import annotations

import json
import os

import pytest

from ddo_eval.benchmarks import mmlu as mmlu_mod
from ddo_eval.benchmarks import mtbench as mtbench_mod
from ddo_eval.benchmarks.mmlu import MMLU_SUBJECTS, _parse_results


def _write_results(tmp_path, payload):
    name = "results_2026.json"
    out = tmp_path / "nested" / "run"
    out.mkdir(parents=True)
    (out / name).write_text(json.dumps(payload))
    return str(tmp_path)


# --- MMLU -------------------------------------------------------------------

def test_all_57_subjects_are_declared():
    assert len(MMLU_SUBJECTS) == 57
    assert len(set(MMLU_SUBJECTS)) == 57


def test_missing_directory_reports_no_score():
    out = _parse_results("/nonexistent/path/xyz")
    assert out["score"] is None
    assert "error" in out


def test_subject_accuracies_are_averaged(tmp_path):
    root = _write_results(tmp_path, {"results": {
        "helm|mmlu:anatomy|5|1": {"em": 0.60},
        "helm|mmlu:astronomy|5|1": {"em": 0.80},
    }})
    out = _parse_results(root)
    assert out["score"] == pytest.approx(0.70)
    assert out["n_subjects"] == 2
    assert len(out["per_subject"]) == 2


def test_exact_match_is_the_reported_metric(tmp_path):
    """Exact match is required even when prefix match is present."""
    root = _write_results(tmp_path, {"results": {
        "helm|mmlu:anatomy|5|1": {"pem": 0.90, "em": 0.10},
    }})
    assert _parse_results(root)["score"] == pytest.approx(0.10)


def test_prefix_match_does_not_substitute_for_exact_match(tmp_path):
    root = _write_results(tmp_path, {"results": {
        "helm|mmlu:anatomy|5|1": {"pem": 0.90},
    }})
    assert _parse_results(root)["score"] is None
    assert _parse_results(root)["metric"] == "em"


def test_mixed_metrics_do_not_produce_an_unlabeled_mean(tmp_path):
    root = _write_results(tmp_path, {"results": {
        "helm|mmlu:a|5|1": {"em": 0.5},
        "helm|mmlu:b|5|1": {"acc,none": 0.7},
        "helm|mmlu:c|5|1": {"acc": 0.9},
    }})
    out = _parse_results(root)
    assert out["score"] is None
    assert set(out["missing_subjects"]) == {"helm|mmlu:b|5|1", "helm|mmlu:c|5|1"}


def test_the_aggregate_row_is_not_counted_as_a_subject(tmp_path):
    root = _write_results(tmp_path, {"results": {
        "all": {"pem": 0.99},
        "helm|mmlu:anatomy|5|1": {"em": 0.40},
    }})
    out = _parse_results(root)
    assert out["n_subjects"] == 1
    assert out["score"] == pytest.approx(0.40)


def test_unrelated_tasks_are_ignored(tmp_path):
    root = _write_results(tmp_path, {"results": {
        "helm|hellaswag|0|0": {"pem": 0.10},
        "helm|mmlu:anatomy|5|1": {"em": 0.50},
    }})
    out = _parse_results(root)
    assert out["n_subjects"] == 1
    assert out["score"] == pytest.approx(0.50)


def test_no_usable_metric_reports_no_score(tmp_path):
    root = _write_results(tmp_path, {"results": {
        "helm|mmlu:anatomy|5|1": {"something_else": 1.0},
    }})
    assert _parse_results(root)["score"] is None


def test_an_empty_results_block_reports_no_score(tmp_path):
    root = _write_results(tmp_path, {"results": {}})
    assert _parse_results(root)["score"] is None


def test_lighteval_must_be_installed(monkeypatch, tmp_path):
    """The pinned version is required, so its absence is a clear error."""
    monkeypatch.setattr(mmlu_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="lighteval"):
        mmlu_mod.run_mmlu("some/model", str(tmp_path))


# --- MT-Bench ---------------------------------------------------------------

def _judge_dir(tmp_path, rows):
    base = tmp_path / "data" / "mt_bench" / "model_judgment"
    base.mkdir(parents=True)
    with open(base / "gpt-4_single.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return tmp_path


def test_missing_judgment_file_reports_no_score(monkeypatch, tmp_path):
    monkeypatch.setattr(mtbench_mod, "_llm_judge_dir", lambda: str(tmp_path))
    out = mtbench_mod._parse_judgments("some_model")
    assert out["score"] is None
    assert "error" in out


def test_scores_are_averaged_for_the_requested_model(monkeypatch, tmp_path):
    _judge_dir(tmp_path, [
        {"model": "mine", "score": 8.0, "category": "writing"},
        {"model": "mine", "score": 6.0, "category": "math"},
    ])
    monkeypatch.setattr(mtbench_mod, "_llm_judge_dir", lambda: str(tmp_path))
    out = mtbench_mod._parse_judgments("mine")
    assert out["score"] == pytest.approx(7.0)
    assert out["n_scored"] == 2


def test_other_models_are_not_mixed_in(monkeypatch, tmp_path):
    """A shared judgment file holds every model that was ever judged."""
    _judge_dir(tmp_path, [
        {"model": "mine", "score": 9.0, "category": "writing"},
        {"model": "someone_else", "score": 1.0, "category": "writing"},
    ])
    monkeypatch.setattr(mtbench_mod, "_llm_judge_dir", lambda: str(tmp_path))
    out = mtbench_mod._parse_judgments("mine")
    assert out["score"] == pytest.approx(9.0)
    assert out["n_scored"] == 1


def test_unjudged_entries_are_discarded(monkeypatch, tmp_path):
    """A score of -1 means the judge produced nothing, not a bad answer."""
    _judge_dir(tmp_path, [
        {"model": "mine", "score": -1, "category": "writing"},
        {"model": "mine", "score": 8.0, "category": "writing"},
    ])
    monkeypatch.setattr(mtbench_mod, "_llm_judge_dir", lambda: str(tmp_path))
    out = mtbench_mod._parse_judgments("mine")
    assert out["n_scored"] == 1
    assert out["score"] == pytest.approx(8.0)


def test_no_matching_model_reports_no_score(monkeypatch, tmp_path):
    _judge_dir(tmp_path, [{"model": "other", "score": 5.0, "category": "writing"}])
    monkeypatch.setattr(mtbench_mod, "_llm_judge_dir", lambda: str(tmp_path))
    assert mtbench_mod._parse_judgments("mine")["score"] is None


def test_no_per_category_breakdown_is_reported(monkeypatch, tmp_path):
    """A judgment record has no category field, so a breakdown cannot be keyed.

    The records carry question_id, model, judge, score and turn; category lives
    in question.jsonl. Keying off entry["category"] put every score into one
    "unknown" bucket, which looked like a breakdown and was not one.
    """
    _judge_dir(tmp_path, [
        {"model": "m", "score": 10.0, "question_id": 81, "turn": 1},
        {"model": "m", "score": 6.0, "question_id": 81, "turn": 2},
    ])
    monkeypatch.setattr(mtbench_mod, "_llm_judge_dir", lambda: str(tmp_path))
    out = mtbench_mod._parse_judgments("m")
    assert out["score"] == pytest.approx(8.0)
    assert "per_category" not in out


def test_a_turn_count_that_does_not_match_is_reported(monkeypatch, tmp_path):
    """gpt-4_single.jsonl is shared and append-only, so the count can differ."""
    _judge_dir(tmp_path, [{"model": "m", "score": 8.0, "question_id": 81, "turn": 1}])
    monkeypatch.setattr(mtbench_mod, "_llm_judge_dir", lambda: str(tmp_path))
    out = mtbench_mod._parse_judgments("m", expected_turns=4)
    assert out["n_scored"] == 1
    assert out["n_turns_expected"] == 4
    assert "expected 4" in out["error"]


def test_missing_data_files_are_reported(monkeypatch, tmp_path):
    """FastChat omits the question set and judge prompts from its package."""
    monkeypatch.setattr(mtbench_mod, "_llm_judge_dir", lambda: str(tmp_path))
    missing = mtbench_mod.check_data_files()
    assert len(missing) == 3
    assert any("question.jsonl" in m for m in missing)
    assert any("judge_prompts.jsonl" in m for m in missing)


def test_judging_requires_an_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        mtbench_mod.run_mtbench("m", str(tmp_path), "mid")


def test_the_standard_question_count_is_declared():
    assert mtbench_mod.N_QUESTIONS == 80


# --- MT-Bench GPU clamp ------------------------------------------------------
# FastChat derives its replica count as num_gpus_total // num_gpus_per_model and
# asserts the division is exact, so clamping the total alone can hand it a value
# that is not a whole multiple.

def test_gpu_clamp_moves_in_whole_replicas(monkeypatch, tmp_path):
    from ddo_eval.benchmarks import mtbench

    captured = {}

    def fake_run(cmd, **kwargs):
        if "gen_model_answer" in " ".join(cmd):
            total = int(cmd[cmd.index("--num-gpus-total") + 1])
            per = int(cmd[cmd.index("--num-gpus-per-model") + 1])
            captured["total"], captured["per"] = total, per
            raise RuntimeError("stop after the command is built")
        raise RuntimeError("unexpected command")

    answer_dir = tmp_path / "data" / "mt_bench" / "model_answer"
    answer_dir.mkdir(parents=True)
    monkeypatch.setattr(mtbench.subprocess, "run", fake_run)
    monkeypatch.setattr(mtbench, "check_data_files", lambda *a, **k: [])
    monkeypatch.setattr(mtbench, "_llm_judge_dir", lambda: str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    with pytest.raises(RuntimeError):
        mtbench.run_mtbench(
            "m", str(tmp_path), model_id="m",
            num_gpus_total=8, num_gpus_per_model=4, n_questions=3,
        )

    assert captured["total"] % captured["per"] == 0, (
        f"--num-gpus-total {captured['total']} is not a multiple of "
        f"--num-gpus-per-model {captured['per']}"
    )


def test_fewer_gpus_than_one_replica_is_reported(monkeypatch, tmp_path):
    from ddo_eval.benchmarks import mtbench

    monkeypatch.setattr(mtbench, "check_data_files", lambda *a, **k: [])
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    with pytest.raises(ValueError, match="not one replica fits"):
        mtbench.run_mtbench("m", str(tmp_path), model_id="m",
                            num_gpus_total=2, num_gpus_per_model=4)


# --- which results file is read ----------------------------------------------
# lighteval writes a new timestamped results_<ts>.json into the same directory on
# every run, so reading whichever the filesystem lists first can report a
# previous run's score.

def test_the_newest_results_file_wins(tmp_path):
    from ddo_eval.benchmarks.mmlu import _parse_results

    old = tmp_path / "results_2024-01-01T00-00-00.json"
    new = tmp_path / "results_2026-01-01T00-00-00.json"
    old.write_text(json.dumps({"results": {"mmlu:abstract_algebra": {"em": 0.10}}}))
    new.write_text(json.dumps({"results": {"mmlu:abstract_algebra": {"em": 0.90}}}))

    # Make the stale file the newer name but the older mtime, so name order and
    # time order disagree and only mtime gives the right answer.
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    out = _parse_results(str(tmp_path))
    assert out["score"] == pytest.approx(0.90)
    assert out["results_file"].endswith(new.name)
