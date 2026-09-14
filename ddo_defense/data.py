"""Fetch and verify the prompt sets, instead of vendoring them.

Nothing is bundled with this library.  Every file is downloaded from a pinned
upstream commit on first use and cached under ``~/.cache/ddo_defense``.  Two
reasons: the splits belong to their original authors and are already published
under a permissive licence, and pinning a commit makes the data a verifiable
input rather than a copy that silently drifts.

The refusal-direction files are checked against recorded git blob hashes.
HarmBench is pinned to a commit and checked for the expected number of standard
behaviours; it does not currently have a recorded file hash.

Sources
-------
``andyrdt/refusal_direction``
    Harmful and harmless instruction splits, and the JailbreakBench prompt set.
    Apache-2.0.
``centerforaisafety/HarmBench``
    Behaviour CSV, from which the 159 standard test behaviours are selected.
    MIT.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

REFUSAL_DIRECTION_REPO = "andyrdt/refusal_direction"
REFUSAL_DIRECTION_COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"

HARMBENCH_REPO = "centerforaisafety/HarmBench"
HARMBENCH_COMMIT = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"

#: ``relative path -> git blob sha1`` at the pinned commit.
_REFUSAL_DIRECTION_FILES: Dict[str, str] = {
    "dataset/splits/harmful_train.json": "5ca6b46750e06bc401ebd93171a6b0dc0590cdd8",
    "dataset/splits/harmful_val.json": "c3128d37e8b255e765f4f51f1a6715572595e3e7",
    "dataset/splits/harmful_test.json": "2dc705dc1a50e7773efca46fedab71229169b3bb",
    "dataset/splits/harmless_train.json": "700a497bd1d20ab074fcc576e9bd79ac604543c5",
    "dataset/splits/harmless_val.json": "6b9ee6e9c789799354b618a046758276da445bd8",
    "dataset/splits/harmless_test.json": "6033b711a3c0bf3d49fb88a4824fdac8be792f25",
    "dataset/processed/jailbreakbench.json": "59e53c47ecd56258160b9937c8ac7af3beb4932d",
}

_HARMBENCH_BEHAVIORS = "data/behavior_datasets/harmbench_behaviors_text_test.csv"

#: HarmBench's standard test subset is exactly this many behaviours.  The count is
#: asserted as a subset check; this is not a content-integrity check.
N_HARMBENCH_STANDARD = 159

SPLITS = ("train", "val", "test")
HARMTYPES = ("harmless", "harmful")


def cache_dir() -> Path:
    """Cache root, overridable with ``DDO_CACHE_DIR``."""
    root = os.environ.get("DDO_CACHE_DIR")
    path = Path(root) if root else Path.home() / ".cache" / "ddo_defense"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git_blob_sha1(data: bytes) -> str:
    """Git's object hash for a blob, so values match ``git hash-object``."""
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def _raw_url(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def _fetch(repo: str, commit: str, path: str, expect_sha: Optional[str] = None) -> Path:
    """Return the cached file, downloading and verifying it if needed."""
    dest = cache_dir() / commit[:12] / path
    if dest.exists():
        if expect_sha is None:
            return dest
        if _git_blob_sha1(dest.read_bytes()) == expect_sha:
            return dest
        dest.unlink()  # Corrupt or truncated; re-fetch below.

    url = _raw_url(repo, commit, path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {path} from {repo}@{commit[:8]}. "
            f"Check network access, or pre-populate {dest}. Original error: {exc}"
        ) from exc

    if expect_sha is not None:
        got = _git_blob_sha1(data)
        if got != expect_sha:
            raise RuntimeError(
                f"Integrity check failed for {path}: expected blob {expect_sha}, "
                f"got {got}. Refusing to use it, since a changed prompt set "
                f"invalidates any measurement made with it."
            )

    dest.write_bytes(data)
    return dest


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_dataset_split(
    harmtype: str,
    split: str,
    instructions_only: bool = False,
) -> List:
    """Load one instruction split.

    Parameters
    ----------
    harmtype
        ``"harmful"`` or ``"harmless"``.
    split
        ``"train"``, ``"val"`` or ``"test"``.
    instructions_only
        Return bare instruction strings rather than full records.
    """
    if harmtype not in HARMTYPES:
        raise ValueError(f"harmtype must be one of {HARMTYPES}, got {harmtype!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    rel = f"dataset/splits/{harmtype}_{split}.json"
    path = _fetch(
        REFUSAL_DIRECTION_REPO, REFUSAL_DIRECTION_COMMIT, rel,
        _REFUSAL_DIRECTION_FILES[rel],
    )
    data = json.loads(path.read_text())
    if instructions_only:
        return [d["instruction"] for d in data]
    return data


def load_dataset(dataset_name: str, instructions_only: bool = False) -> List:
    """Load a processed benchmark set.  Currently ``"jailbreakbench"``."""
    rel = f"dataset/processed/{dataset_name}.json"
    if rel not in _REFUSAL_DIRECTION_FILES:
        available = sorted(
            k.split("/")[-1].removesuffix(".json")
            for k in _REFUSAL_DIRECTION_FILES
            if k.startswith("dataset/processed/")
        )
        raise ValueError(f"Unknown dataset {dataset_name!r}. Available: {available}")

    path = _fetch(
        REFUSAL_DIRECTION_REPO, REFUSAL_DIRECTION_COMMIT, rel,
        _REFUSAL_DIRECTION_FILES[rel],
    )
    data = json.loads(path.read_text())
    if instructions_only:
        return [d["instruction"] for d in data]
    return data


def load_harmbench_standard() -> List[Dict[str, str]]:
    """The 159 standard HarmBench test behaviours, as ``{"instruction": ...}``.

    Selection filters the behaviour CSV on ``FunctionalCategory == "standard"``,
    which is reproducible from the pinned file alone and needs no extra id list.
    """
    path = _fetch(HARMBENCH_REPO, HARMBENCH_COMMIT, _HARMBENCH_BEHAVIORS)

    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("FunctionalCategory") or "").strip().lower() == "standard":
                rows.append({
                    "instruction": row["Behavior"],
                    "behavior_id": row.get("BehaviorID", ""),
                    "category": row.get("SemanticCategory", ""),
                })

    if len(rows) != N_HARMBENCH_STANDARD:
        raise RuntimeError(
            f"Expected {N_HARMBENCH_STANDARD} standard HarmBench behaviours, "
            f"found {len(rows)}. The CSV is pinned to a commit, so the likely "
            f"causes are a corrupt download or an upstream schema change in "
            f"the FunctionalCategory column."
        )
    return rows


def load_eval_set(name: str = "jailbreakbench") -> List[Dict[str, str]]:
    """Load an attack evaluation set by name.

    ``"jailbreakbench"`` gives 100 behaviours and ``"harmbench_standard"`` 159;
    these are the attack test sets.  ``"harmful_val"`` is the held-out
    validation split, which is what hyperparameter search scores on so that
    selection never happens on a reported benchmark.
    """
    if name == "jailbreakbench":
        return load_dataset("jailbreakbench")
    if name == "harmbench_standard":
        return load_harmbench_standard()
    if name in ("harmful_val", "harmful_test"):
        split = name.split("_", 1)[1]
        return load_dataset_split("harmful", split)
    raise ValueError(
        f"Unknown eval set {name!r}. Use 'jailbreakbench', 'harmbench_standard', "
        f"'harmful_val' or 'harmful_test'."
    )


def prefetch() -> List[Path]:
    """Fetch and verify the pinned files now, so later runs need no network.

    Covers the seven refusal-direction splits and the HarmBench behaviour CSV.
    XSTest is pulled from the Hugging Face hub on demand and the MT-Bench data
    files from FastChat, so neither is prefetched here.
    """
    paths = [
        _fetch(REFUSAL_DIRECTION_REPO, REFUSAL_DIRECTION_COMMIT, rel, sha)
        for rel, sha in _REFUSAL_DIRECTION_FILES.items()
    ]
    paths.append(_fetch(HARMBENCH_REPO, HARMBENCH_COMMIT, _HARMBENCH_BEHAVIORS))
    return paths
