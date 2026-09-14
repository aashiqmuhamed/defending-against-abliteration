"""The tuner's decision logic, which is what makes a search trustworthy.

Two properties matter here and neither needs a GPU. The search space must be
seedable from a measurement, so an unseen model starts somewhere sensible rather
than at a generic default. And a trial that fails a gate must be unable to win:
the rejection value has to sit above any achievable attack success rate, or a
broken checkpoint could come out on top of a minimisation.
"""

from __future__ import annotations


from ddo_defense.tuning import (
    REJECTED_TRIAL_VALUE,
    SearchSpace,
    TrialResult,
    _suggest_params,
)


class StubTrial:
    """Records what was asked for and answers with the low end of each range."""

    def __init__(self, categorical_index: int = 0):
        self.number = 0
        self.asked: dict = {}
        self._categorical_index = categorical_index

    def suggest_int(self, name, low, high):
        self.asked[name] = ("int", low, high)
        return low

    def suggest_float(self, name, low, high, log=False):
        self.asked[name] = ("float", low, high, log)
        return low

    def suggest_categorical(self, name, choices):
        self.asked[name] = ("categorical", tuple(choices))
        return choices[self._categorical_index]


def test_rejection_value_cannot_win_a_minimisation():
    """Attack success is a rate in [0, 1]; rejection must be far above it."""
    assert REJECTED_TRIAL_VALUE > 1.0


def test_a_rejected_trial_carries_its_reason():
    result = TrialResult(params={}, value=REJECTED_TRIAL_VALUE,
                         rejection_reason="coherence: too short")
    assert result.value > 1.0
    assert result.rejection_reason
    assert not result.passed_coherence


def test_default_space_samples_every_parameter():
    space = SearchSpace()
    trial = StubTrial()
    params = _suggest_params(trial, space)

    for name in ("layer_start", "layer_end", "init_beta", "init_scale",
                 "confusion_lambda", "lr", "epochs", "compile_mode"):
        assert name in params, name
    assert "n_decoys" in params


def test_learning_rate_is_sampled_logarithmically():
    """Spanning 1e-3 to 5e-2 linearly would barely visit the small end."""
    trial = StubTrial()
    _suggest_params(trial, SearchSpace())
    assert trial.asked["lr"][0] == "float"
    assert trial.asked["lr"][3] is True


def test_sampled_values_sit_inside_the_space():
    space = SearchSpace()
    params = _suggest_params(StubTrial(), space)
    assert space.layer_start[0] <= params["layer_start"] <= space.layer_start[1]
    assert space.layer_end[0] <= params["layer_end"] <= space.layer_end[1]
    assert space.init_beta[0] <= params["init_beta"] <= space.init_beta[1]
    assert params["compile_mode"] in space.compile_modes


def test_fixed_n_decoys_is_not_offered_to_the_sampler():
    """A degenerate range should not consume a search dimension."""
    trial = StubTrial()
    params = _suggest_params(trial, SearchSpace(n_decoys=(1, 1)))
    assert params["n_decoys"] == 1
    assert "n_decoys" not in trial.asked


def test_a_real_n_decoys_range_is_sampled():
    trial = StubTrial()
    params = _suggest_params(trial, SearchSpace(n_decoys=(1, 4)))
    assert "n_decoys" in trial.asked
    assert 1 <= params["n_decoys"] <= 4


def test_space_is_seeded_from_a_fingerprint_suggestion():
    """The measured band becomes the centre of the search, not a fixed choice."""
    suggestion = {
        "search_layer_start": (2, 6),
        "search_layer_end": (10, 20),
        "compile_mode": "additive",
    }
    space = SearchSpace.from_suggestion(suggestion)
    assert space.layer_start == (2, 6)
    assert space.layer_end == (10, 20)
    # The suggested mode is tried first, without excluding the alternative.
    assert space.compile_modes[0] == "additive"
    assert set(space.compile_modes) == {"additive", "replace"}


def test_a_replace_suggestion_still_keeps_additive_available():
    space = SearchSpace.from_suggestion({"compile_mode": "replace"})
    assert space.compile_modes[0] == "replace"
    assert "additive" in space.compile_modes


def test_an_empty_suggestion_leaves_the_defaults():
    default = SearchSpace()
    seeded = SearchSpace.from_suggestion({})
    assert seeded.layer_start == default.layer_start
    assert seeded.layer_end == default.layer_end


def test_a_nonsense_compile_mode_is_ignored():
    """A bad hint must not smuggle an invalid mode into the search."""
    space = SearchSpace.from_suggestion({"compile_mode": "soft"})
    assert set(space.compile_modes) <= {"replace", "additive"}


def test_suggestion_from_the_real_fingerprint_path_is_accepted():
    """End to end: suggest_config output must be consumable as a search space."""
    from ddo_defense.fingerprint import suggest_config

    fp = {
        "zone_start": 8, "zone_coverage": 0.4, "gate_arch": "geglu",
        "rfa_compliance": 0.8, "rfa_degenerate": 0.0, "d0_to_d60_ratio": 2.0,
        "architecture": {"n_layers": 32},
    }
    space = SearchSpace.from_suggestion(suggest_config(fp))
    params = _suggest_params(StubTrial(), space)
    assert params["layer_start"] < params["layer_end"]
    assert params["compile_mode"] == "additive"


# --- the search space has to be able to reach the reference configurations ---
# Bands below are written as inclusive first/last layers; the code's layer_end is
# exclusive, so L6--14 is layer_start=6, layer_end=15.

REFERENCE_CONFIGS = {
    "Yi":       dict(first=18, last=32, beta=6.1,  scale=0.42, conf=0.5,  lr=0.002, ep=2),
    "Llama-2":  dict(first=4,  last=12, beta=2.9,  scale=0.47, conf=1.1,  lr=0.005, ep=3),
    "Llama-3":  dict(first=6,  last=14, beta=1.30, scale=0.46, conf=1.12, lr=0.003, ep=2),
    "Gemma-2":  dict(first=10, last=18, beta=6.42, scale=0.36, conf=1.79, lr=0.021, ep=1),
    "Qwen3":    dict(first=12, last=16, beta=3.20, scale=0.08, conf=1.79, lr=0.012, ep=3),
    "Mistral":  dict(first=2,  last=18, beta=4.05, scale=0.46, conf=0.82, lr=0.008, ep=2),
    "GLM-4":    dict(first=6,  last=17, beta=5.3,  scale=0.29, conf=1.2,  lr=0.016, ep=1),
}


def _contains(rng, value):
    return rng[0] <= value <= rng[1]


def test_default_space_can_reach_every_reference_config():
    from ddo_defense.tuning import SearchSpace

    space = SearchSpace()
    unreachable = []
    for name, cfg in REFERENCE_CONFIGS.items():
        checks = {
            "layer_start": _contains(space.layer_start, cfg["first"]),
            "layer_end": _contains(space.layer_end, cfg["last"] + 1),
            "init_beta": _contains(space.init_beta, cfg["beta"]),
            "init_scale": _contains(space.init_scale, cfg["scale"]),
            "confusion_lambda": _contains(space.confusion_lambda, cfg["conf"]),
            "lr": _contains(space.lr, cfg["lr"]),
            "epochs": _contains(space.epochs, cfg["ep"]),
        }
        missing = [k for k, ok in checks.items() if not ok]
        if missing:
            unreachable.append(f"{name}: {', '.join(missing)}")
    assert not unreachable, "search space cannot produce: " + "; ".join(unreachable)


def test_sampled_band_is_never_empty():
    """layer_start and layer_end are drawn from overlapping ranges.

    An independent draw could put the end at or below the start, and such a band
    cannot be applied at all, so every sample has to leave at least one layer.
    """
    from ddo_defense.tuning import SearchSpace, _suggest_params

    space = SearchSpace()
    # The worst case for an independent draw: the highest start with the lowest end.
    assert space.layer_start[1] >= space.layer_end[0], (
        "ranges no longer overlap, so this test guards nothing"
    )

    class HighStartLowEnd(StubTrial):
        def suggest_int(self, name, low, high):
            self.asked[name] = ("int", low, high)
            return high if name == "layer_start" else low

    params = _suggest_params(HighStartLowEnd(), space)
    assert params["layer_start"] < params["layer_end"]
    assert params["layer_start"] == space.layer_start[1]


def test_both_compile_modes_are_searched():
    from ddo_defense.tuning import SearchSpace

    assert set(SearchSpace().compile_modes) == {"replace", "additive"}


def test_rank_is_one_by_default():
    from ddo_defense.tuning import SearchSpace

    assert SearchSpace().n_decoys == (1, 1)


def test_best_params_carries_every_applied_parameter():
    """Optuna records only sampled values; a config file needs all of them.

    ``n_decoys`` is fixed rather than searched, so it never passes through
    ``trial.suggest_*`` and is absent from ``trial.params``.  Reporting that as
    the tuned configuration would let ``ddo-apply`` silently substitute its own
    default.
    """
    space = SearchSpace()
    trial = StubTrial()
    params = _suggest_params(trial, space)

    assert "n_decoys" not in trial.asked, "fixed value should not be sampled"
    assert params["n_decoys"] == 1, "but it must still appear in the parameters"

    from ddo_defense.cli import CONFIG_KEYS
    missing = [k for k in CONFIG_KEYS if k not in params and k not in ("n_readers", "reader_gamma")]
    assert not missing, f"a tuned config would omit {missing}"


# --- the gates the module's central claim rests on ---------------------------
# "Trials failing either gate are rejected with a large penalty rather than
# scored, so a failure can never win." That had no test.

class _StubAdapter:
    def __init__(self, responses):
        self._responses = responses
        self.model = object()
        self.tokenizer = object()

    def generate(self, prompts, **kwargs):
        return [self._responses] * len(prompts)

    def free(self):
        pass


def test_a_trial_failing_coherence_cannot_win(monkeypatch):
    from ddo_defense import tuning as t

    class _Report:
        passed = False
        failures = ["empty output"]

    monkeypatch.setattr(t, "REJECTED_TRIAL_VALUE", 100.0, raising=False)
    monkeypatch.setattr("ddo_defense.coherence.coherence_gate", lambda *a, **k: _Report())

    out = t.evaluate_trial(_StubAdapter("x"), judges=("substring",))
    assert out.value == t.REJECTED_TRIAL_VALUE
    assert out.passed_coherence is False
    assert "coherence" in out.rejection_reason
    assert out.value > 1.0, "a rejection must sit above any achievable ASR"


def test_a_trial_below_the_benign_floor_cannot_win(monkeypatch):
    from ddo_defense import tuning as t

    class _Report:
        passed = True
        failures = []

        def summary(self):
            return "ok"

    monkeypatch.setattr("ddo_defense.coherence.coherence_gate", lambda *a, **k: _Report())

    # Every benign prompt is refused, so compliance is 0 and the floor rejects.
    out = t.evaluate_trial(
        _StubAdapter("I cannot help with that."),
        judges=("substring",),
        benign_prompts=["p1", "p2", "p3"],
        xstest_floor=0.85,
    )
    assert out.value == t.REJECTED_TRIAL_VALUE
    assert out.passed_coherence is True
    assert out.passed_floor is False
    assert out.benign_compliance == 0.0
    assert "below floor" in out.rejection_reason


def test_the_rejection_value_beats_any_real_attack_success():
    """A rejected trial must lose to a trial that scored 100% attack success."""
    from ddo_defense.tuning import REJECTED_TRIAL_VALUE

    assert REJECTED_TRIAL_VALUE > 1.0
