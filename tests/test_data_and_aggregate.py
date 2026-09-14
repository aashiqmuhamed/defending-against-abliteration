"""Data integrity hashing, and the cross-judge mean."""

from __future__ import annotations

import json
import subprocess

import pytest

from ddo_defense.data import _git_blob_sha1
from ddo_eval.aggregate import aggregate_results, format_table, mean_asr


def test_blob_hash_matches_git():
    """Our hash must equal `git hash-object`, or pins cannot be checked."""
    data = b"hello ddo\n"
    ours = _git_blob_sha1(data)
    try:
        theirs = subprocess.run(
            ["git", "hash-object", "--stdin"], input=data,
            capture_output=True, check=True,
        ).stdout.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git unavailable")
    assert ours == theirs


def test_mean_ignores_the_avg_key():
    assert mean_asr({"a": 0.2, "b": 0.4, "avg": 99.0}, required_judges=("a", "b")) == pytest.approx(0.3)


def test_mean_of_nothing_is_none_not_zero():
    """A missing measurement must not read as a perfect defense."""
    assert mean_asr({}) is None
    assert mean_asr({"a": None}) is None


def test_aggregate_computes_three_judge_mean(tmp_path):
    model_dir = tmp_path / "model_a"
    model_dir.mkdir()
    (model_dir / "mmlu.json").write_text(json.dumps({"score": 0.668}))
    (model_dir / "xstest.json").write_text(json.dumps({"score": 0.92}))
    for judge, asr in (("harmbench_cls", 0.03), ("llamaguard2", 0.01), ("strongreject", 0.02)):
        (model_dir / f"scores_{judge}_attack_RFA.json").write_text(json.dumps({"asr": asr}))

    table = aggregate_results(str(tmp_path))
    row = table["models"]["model_a"]

    assert row["capability"]["mmlu"] == pytest.approx(0.668)
    assert sorted(table["judges"]) == ["harmbench_cls", "llamaguard2", "strongreject"]
    assert row["asr"]["RFA"]["avg"] == pytest.approx(0.02)
    # Per-judge numbers survive alongside the mean.
    assert row["asr"]["RFA"]["llamaguard2"] == pytest.approx(0.01)
    assert "model_a" in format_table(table)


def test_legacy_subset_is_incomplete_without_a_protocol_manifest(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "scores_llamaguard2_attack_RFA.json").write_text(json.dumps({"asr": 0.5}))
    table = aggregate_results(str(tmp_path))
    assert table["models"]["m"]["asr"]["RFA"]["avg"] is None
    assert "incomplete" in format_table(table)


def test_missing_results_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate_results(str(tmp_path / "nope"))


# --- the in-process aggregation entry point ---------------------------------

def test_aggregate_from_scores_adds_the_mean_per_attack():
    from ddo_eval.aggregate import aggregate_from_scores

    row = aggregate_from_scores(
        {
            "RFA": {"harmbench_cls": 0.02, "llamaguard2": 0.04, "strongreject": 0.06},
            "DirectRequest": {"llamaguard2": 0.10},
        },
        capability={"mmlu": 0.668},
    )

    assert row["capability"] == {"mmlu": 0.668}
    assert row["asr"]["RFA"]["avg"] == pytest.approx(0.04)
    assert row["asr"]["RFA"]["llamaguard2"] == pytest.approx(0.04)
    assert row["asr"]["DirectRequest"]["avg"] is None


def test_aggregate_from_scores_reports_none_when_no_judge_scored():
    from ddo_eval.aggregate import aggregate_from_scores

    row = aggregate_from_scores({"RFA": {}})
    assert row["asr"]["RFA"]["avg"] is None
    assert row["capability"] == {}


# --- the integrity check that is this module's reason to exist ----------------

def test_a_tampered_cache_is_refused(tmp_path, monkeypatch):
    """A changed prompt set invalidates every number measured with it."""
    from ddo_defense import data

    monkeypatch.setenv("DDO_CACHE_DIR", str(tmp_path))
    rel = "dataset/splits/harmful_val.json"
    dest = tmp_path / data.REFUSAL_DIRECTION_COMMIT[:12] / rel
    dest.parent.mkdir(parents=True)
    dest.write_text('[{"instruction": "tampered", "category": "x"}]')

    # A cached file whose hash does not match is deleted and re-fetched, so block
    # the network to prove the mismatch is what stops it.
    def no_network(*a, **k):
        raise OSError("network disabled for this test")

    monkeypatch.setattr(data.urllib.request, "urlopen", no_network)
    with pytest.raises(RuntimeError, match="Could not download"):
        data.load_dataset_split("harmful", "val")


def test_a_download_whose_hash_is_wrong_is_refused(tmp_path, monkeypatch):
    from ddo_defense import data

    monkeypatch.setenv("DDO_CACHE_DIR", str(tmp_path))

    class _Resp:
        def read(self):
            return b'[{"instruction": "not the pinned content"}]'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="Integrity check failed"):
        data.load_dataset_split("harmful", "val")


def test_a_matching_hash_is_accepted(tmp_path, monkeypatch):
    from ddo_defense import data

    monkeypatch.setenv("DDO_CACHE_DIR", str(tmp_path))
    payload = b'[{"instruction": "x"}]'
    sha = data._git_blob_sha1(payload)
    monkeypatch.setitem(data._REFUSAL_DIRECTION_FILES, "dataset/splits/harmful_val.json", sha)

    class _Resp:
        def read(self):
            return payload
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert data.load_dataset_split("harmful", "val") == [{"instruction": "x"}]
