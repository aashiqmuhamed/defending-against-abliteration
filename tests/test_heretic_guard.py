"""The Heretic environment guard, which is the first thing a user meets.

Heretic pins a different transformers major version, so it cannot share this
environment and has to run from its own. That makes the missing-environment error
the most likely thing a user encounters, and it has to carry the setup steps
rather than a bare import failure.
"""

from __future__ import annotations

import os

import pytest

from ddo_defense.heretic import (
    DEFAULT_VENV,
    HERETIC_PACKAGE,
    HERETIC_VARIANTS,
    check_environment,
    venv_python,
)


def test_interpreter_path_is_inside_the_venv():
    assert venv_python("some_env") == os.path.join("some_env", "bin", "python")
    assert venv_python() == os.path.join(DEFAULT_VENV, "bin", "python")


def test_a_missing_environment_is_reported_with_setup_steps(tmp_path):
    missing = str(tmp_path / "no_such_env")
    with pytest.raises(RuntimeError) as excinfo:
        check_environment(missing)

    message = str(excinfo.value)
    assert missing in message
    # The message must be actionable, not just a complaint.
    assert "python -m venv" in message
    assert HERETIC_PACKAGE in message


def test_the_error_explains_why_it_must_stay_separate(tmp_path):
    with pytest.raises(RuntimeError, match="transformers"):
        check_environment(str(tmp_path / "absent"))


def test_the_package_pin_is_explicit():
    """An unpinned install would pull a version with a different interface."""
    assert "==" in HERETIC_PACKAGE
    assert HERETIC_PACKAGE.startswith("heretic-llm==")


def test_both_documented_variants_are_declared():
    assert set(HERETIC_VARIANTS) == {"Heretic", "Heretic_harmbench"}
    assert HERETIC_VARIANTS["Heretic"]["eval_set"] == "jailbreakbench"
    assert HERETIC_VARIANTS["Heretic_harmbench"]["eval_set"] == "harmbench_standard"
