"""Heretic: an optimizing abliteration attack, run from a separate environment.

Heretic searches ablation parameters with Optuna rather than removing a single
estimated direction, so it is a stronger attack than plain refusal feature
ablation and worth reporting separately.

It cannot share this environment.  Heretic pins a different transformers major
version, and installing it alongside this library breaks one or the other, so it
runs from its own virtual environment.

**Heretic is an interactive tool.**  Every released version, 1.2.0 through 1.4.0,
finishes its Optuna search and then asks a human which trial to keep
(``prompt_select``) and where to write it (``prompt_path``).  There is no
settings field for an output directory -- the only path-shaped setting is
``residual_plot_path`` -- so a subprocess cannot drive it to a saved checkpoint.

The split here follows from that: :func:`heretic_command` prints the command to
run yourself, and :func:`evaluate_heretic_checkpoint` takes the folder you saved
and produces completions for judging.

Setup, once:

.. code-block:: bash

    python -m venv heretic_venv
    heretic_venv/bin/pip install 'heretic-llm==1.2.0'

The version is pinned because later releases changed both the settings interface
and the transformers requirement.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence

HERETIC_PACKAGE = "heretic-llm==1.2.0"
DEFAULT_VENV = "heretic_venv"

#: Reported variants: the same optimization against two evaluation sets.
HERETIC_VARIANTS: Dict[str, Dict[str, str]] = {
    "Heretic": {"eval_set": "jailbreakbench"},
    "Heretic_harmbench": {"eval_set": "harmbench_standard"},
}


def venv_python(venv: str = DEFAULT_VENV) -> str:
    """Path to the interpreter inside the Heretic environment."""
    return os.path.join(venv, "bin", "python")


def check_environment(venv: str = DEFAULT_VENV) -> None:
    """Raise with setup instructions unless the Heretic environment is usable."""
    python = venv_python(venv)
    if not os.path.exists(python):
        raise RuntimeError(
            f"No Heretic environment at {venv!r}. Create one with:\n"
            f"  python -m venv {venv}\n"
            f"  {venv}/bin/pip install '{HERETIC_PACKAGE}'\n"
            f"It must stay separate: Heretic pins a different transformers "
            f"version than this library."
        )
    probe = subprocess.run(
        [python, "-c", "import heretic; print(getattr(heretic, '__version__', 'unknown'))"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"The environment at {venv!r} exists but cannot import heretic "
            f"({(probe.stderr or '').strip()[-200:] or 'no error output'}). "
            f"Install it with: {venv}/bin/pip install '{HERETIC_PACKAGE}'"
        )


def heretic_command(
    model_path: str,
    *,
    venv: str = DEFAULT_VENV,
    n_trials: int = 200,
    n_startup_trials: int = 15,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """The command to run Heretic against ``model_path``, as argv.

    Run it yourself in a terminal.  It cannot be driven from a subprocess: when
    the search finishes, Heretic asks which trial to keep and where to save it,
    and neither answer is expressible as a setting.

    The flag names come from Heretic's pydantic-settings ``Settings`` model, so
    they are the field names: ``--n_trials``, not ``--trials``.  The entry point
    is the ``heretic`` console script; the package ships no ``__main__``, so
    ``python -m heretic`` does not work.
    """
    check_environment(venv)
    cmd = [
        os.path.join(venv, "bin", "heretic"),
        "--model", model_path,
        "--n_trials", str(n_trials),
        "--n_startup_trials", str(n_startup_trials),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def print_heretic_instructions(
    model_path: str,
    *,
    venv: str = DEFAULT_VENV,
    n_trials: int = 200,
    n_startup_trials: int = 15,
) -> List[str]:
    """Print what to run and what to do with the result."""
    cmd = heretic_command(
        model_path, venv=venv, n_trials=n_trials, n_startup_trials=n_startup_trials
    )
    print("Run this in a terminal:\n")
    print("  " + " ".join(cmd) + "\n")
    print(
        "When the search finishes Heretic will ask which trial to keep and for a\n"
        "folder to write it to. Pick a trial, give it a path, then pass that path\n"
        "to evaluate_heretic_checkpoint() to generate completions for judging."
    )
    return cmd


def evaluate_heretic_checkpoint(
    abliterated: str,
    *,
    eval_set: str = "jailbreakbench",
    n_trials: int = 200,
    n_startup_trials: int = 15,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    n_eval: Optional[int] = None,
    output_dir: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """Generate completions from a checkpoint Heretic already produced.

    ``abliterated`` is the folder you gave Heretic when it asked where to save.
    ``n_trials`` and ``n_startup_trials`` are recorded in the result so the
    number can be reported with the budget that produced it; they do not drive
    anything here.
    """
    from ddo_defense.data import load_eval_set
    from ddo_defense.models import ModelAdapter

    if not os.path.isdir(abliterated):
        raise FileNotFoundError(
            f"No checkpoint at {abliterated!r}. Run heretic_command() output in a "
            f"terminal first, save a trial when it asks, and pass that folder here."
        )

    meta: Dict[str, Any] = {}
    meta_path = os.path.join(abliterated, "heretic_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)

    eval_data = load_eval_set(eval_set)
    if n_eval:
        eval_data = eval_data[:n_eval]

    adapter = ModelAdapter.from_pretrained(
        abliterated, trust_remote_code=trust_remote_code
    )
    try:
        prompts = [row["instruction"] for row in eval_data]
        responses = adapter.generate(
            prompts, max_new_tokens=max_new_tokens, batch_size=batch_size
        )
        completions = [
            {"prompt": p, "response": r} for p, r in zip(prompts, responses)
        ]
    finally:
        adapter.free()

    # The variant name has to follow the eval set, or both runs write the same
    # judging_Heretic.json and the second overwrites the first.
    attack_name = next(
        (name for name, cfg in HERETIC_VARIANTS.items() if cfg["eval_set"] == eval_set),
        "Heretic",
    )

    result = {
        "attack": attack_name,
        "eval_set": eval_set,
        "n_trials": n_trials,
        "n_startup_trials": n_startup_trials,
        "n_completions": len(completions),
        "completions": completions,
        "heretic_meta": meta,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"attack_{attack_name}.json"), "w") as fh:
            json.dump(result, fh, indent=2)
    return result
