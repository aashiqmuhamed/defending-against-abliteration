"""Configuration precedence for ``ddo-apply``.

The documented path is: start from the built-in defaults, load a tuned config,
then let explicit flags win. That only works if an unsupplied flag is
distinguishable from a supplied one, which means every tunable flag must default
to ``None``. A flag with a real default looks supplied on every run and silently
discards what the tuned config said, which would quietly undo the tuning the user
just paid for.
"""

from __future__ import annotations

import json

import pytest


from ddo_defense.cli import CONFIG_KEYS, DEFAULT_LLAMA3_CONFIG, merge_config


def test_defaults_alone():
    params = merge_config(DEFAULT_LLAMA3_CONFIG)
    assert params == DEFAULT_LLAMA3_CONFIG
    assert params is not DEFAULT_LLAMA3_CONFIG  # must not be mutated in place


def test_defaults_are_not_mutated():
    before = dict(DEFAULT_LLAMA3_CONFIG)
    merge_config(DEFAULT_LLAMA3_CONFIG, {"layer_start": 99}, {"epochs": 7})
    assert DEFAULT_LLAMA3_CONFIG == before


def test_a_config_file_beats_the_defaults():
    params = merge_config(DEFAULT_LLAMA3_CONFIG, {"layer_start": 20, "epochs": 3})
    assert params["layer_start"] == 20
    assert params["epochs"] == 3
    # Untouched keys survive.
    assert params["init_beta"] == DEFAULT_LLAMA3_CONFIG["init_beta"]


def test_a_tuner_result_is_unwrapped():
    payload = {"best_params": {"layer_start": 11}, "best_asr": 0.02, "seed": 42}
    params = merge_config(DEFAULT_LLAMA3_CONFIG, payload)
    assert params["layer_start"] == 11
    # Metadata from the tuner must not leak into the parameters.
    assert "best_asr" not in params
    assert "seed" not in params


def test_an_explicit_flag_beats_the_config_file():
    params = merge_config(
        DEFAULT_LLAMA3_CONFIG,
        {"layer_start": 20, "compile_mode": "replace"},
        {"layer_start": 4, "compile_mode": "additive"},
    )
    assert params["layer_start"] == 4
    assert params["compile_mode"] == "additive"


def test_unsupplied_flags_are_ignored():
    params = merge_config(
        DEFAULT_LLAMA3_CONFIG,
        {"layer_start": 20},
        {key: None for key in CONFIG_KEYS},
    )
    assert params["layer_start"] == 20


def test_a_tuned_n_decoys_survives_when_the_flag_is_absent():
    """A flag carrying a real default would reset this silently, so it is None."""
    params = merge_config(
        DEFAULT_LLAMA3_CONFIG,
        {"best_params": {"n_decoys": 4}},
        {"n_decoys": None},
    )
    assert params["n_decoys"] == 4


def test_n_decoys_can_still_be_overridden_deliberately():
    params = merge_config(
        DEFAULT_LLAMA3_CONFIG, {"best_params": {"n_decoys": 4}}, {"n_decoys": 2}
    )
    assert params["n_decoys"] == 2


def test_every_tunable_flag_parses_to_none_when_omitted():
    """A flag with a real default would silently override --config every run."""
    from ddo_defense.cli import CONFIG_KEYS, build_apply_parser

    args = build_apply_parser().parse_args(["--model_path", "x"])
    for key in CONFIG_KEYS:
        assert getattr(args, key) is None, f"--{key} carries a default"


def test_every_config_key_has_a_flag():
    from ddo_defense.cli import CONFIG_KEYS, build_apply_parser

    flags = {
        opt.lstrip("-")
        for action in build_apply_parser()._actions
        for opt in action.option_strings
    }
    for key in CONFIG_KEYS:
        assert key in flags, f"{key} is in CONFIG_KEYS but has no flag"


def test_every_config_key_is_overridable():
    overrides = {
        "layer_start": 1, "layer_end": 2, "init_beta": 9.0, "init_scale": 0.9,
        "confusion_lambda": 0.5, "lr": 0.05, "epochs": 1,
        "compile_mode": "additive", "n_decoys": 3,
    }
    params = merge_config(DEFAULT_LLAMA3_CONFIG, None, overrides)
    for key, value in overrides.items():
        assert params[key] == value


def test_a_flat_config_without_best_params_works(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"layer_start": 7, "layer_end": 16}))
    payload = json.loads(path.read_text())
    params = merge_config(DEFAULT_LLAMA3_CONFIG, payload)
    assert params["layer_start"] == 7
    assert params["layer_end"] == 16


def test_builtin_defaults_cover_every_key():
    assert set(DEFAULT_LLAMA3_CONFIG) == set(CONFIG_KEYS)


# --- a tuner result that found nothing ---------------------------------------
# tune() always writes its output file, and writes best_params: null when no
# trial passed both gates. Applying that file must say so, not fall through to
# the built-in defaults under the name of a tuned configuration.

def test_a_tuner_result_with_no_accepted_trial_is_refused():
    payload = {"best_params": None, "best_asr": None, "n_accepted": 0, "n_trials": 30}
    with pytest.raises(ValueError, match="best_params: null"):
        merge_config(DEFAULT_LLAMA3_CONFIG, payload)


def test_an_empty_best_params_is_refused_too():
    with pytest.raises(ValueError, match="nothing to apply"):
        merge_config(DEFAULT_LLAMA3_CONFIG, {"best_params": {}})


def test_a_populated_best_params_still_applies():
    params = merge_config(DEFAULT_LLAMA3_CONFIG, {"best_params": {"epochs": 3}})
    assert params["epochs"] == 3
    assert params["init_beta"] == DEFAULT_LLAMA3_CONFIG["init_beta"]
