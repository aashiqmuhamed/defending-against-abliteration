"""Command line entry points.

Three commands, in the order you would normally use them:

``ddo-fingerprint``
    Measure where refusal lives in a model and get a suggested layer band.
``ddo-tune``
    Search for a configuration that defends the model without breaking it.
``ddo-apply``
    Apply a known configuration and save the defended checkpoint.

Every command works from a clone with no install, as
``python -m ddo_defense.cli ...``, and as a console script after
``pip install -e .``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence


#: Built-in Llama-3 configuration.  It is the base of every merge, not only the
#: no-config case: any key a tuned file omits is filled from here, so a config
#: missing a key gets Llama-3's value for it rather than an error.
DEFAULT_LLAMA3_CONFIG = {
    "layer_start": 6, "layer_end": 15, "init_beta": 1.3, "init_scale": 0.463,
    "confusion_lambda": 1.12, "lr": 0.0026, "epochs": 2,
    "compile_mode": "replace", "n_decoys": 1,
    "n_readers": 1, "reader_gamma": 0.3,
}

#: Keys a tuned config may carry and that flags may override.
CONFIG_KEYS = (
    "layer_start", "layer_end", "init_beta", "init_scale",
    "confusion_lambda", "lr", "epochs", "compile_mode", "n_decoys",
    "n_readers", "reader_gamma",
)


def merge_config(defaults, config_payload=None, overrides=None):
    """Resolve a configuration: defaults, then a tuned file, then explicit flags.

    A flag counts as supplied only when it is not ``None``. Every tunable flag
    must therefore declare ``default=None``: a flag carrying a real default would
    look supplied on every run and silently overwrite whatever the tuned config
    said, which is the opposite of what ``--config`` is for.

    ``config_payload`` may be a tuner result, in which case its ``best_params``
    are used, or a flat mapping of parameters.  A tuner result whose search found
    nothing carries ``best_params: null``; that is reported rather than applied,
    because the alternative is deploying the built-in defaults under the name of
    a tuned configuration.
    """
    params = dict(defaults)
    if config_payload:
        if "best_params" in config_payload:
            best = config_payload["best_params"]
            if not best:
                raise ValueError(
                    "This tuner result has best_params: null, so no trial passed "
                    "both gates and there is nothing to apply. Re-run ddo-tune, "
                    "widening the search or lowering --xstest_floor."
                )
        else:
            best = config_payload
        params.update(best)
    overrides = overrides or {}
    for key in CONFIG_KEYS:
        if overrides.get(key) is not None:
            params[key] = overrides[key]
    return params


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path", required=True, help="Checkpoint to work on")
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Execute model code from the checkpoint. Needed by a few "
             "checkpoints; off by default because it runs third-party code.",
    )


def _load_prompts(n: int):
    from ddo_defense.data import load_dataset_split

    harmful = load_dataset_split("harmful", "train", instructions_only=True)[:n]
    harmless = load_dataset_split("harmless", "train", instructions_only=True)[:n]
    return harmful, harmless


def _write_json(obj, path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- fingerprint

def fingerprint_main(argv: Optional[Sequence[str]] = None) -> int:
    """Measure a model and print a suggested starting configuration."""
    parser = argparse.ArgumentParser(
        prog="ddo-fingerprint",
        description="Measure where refusal lives in a model, and suggest a DDO "
                    "layer band. Runtime is dominated by the per-layer generation sweep.",
    )
    _add_model_args(parser)
    parser.add_argument("--n_harmful", type=int, default=16,
                        help="Probe prompts per measurement (default 16)")
    parser.add_argument("--n_probes", type=int, default=128,
                        help="Prompts for estimating the refusal direction")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Write the fingerprint as JSON")
    args = parser.parse_args(argv)

    from ddo_defense.directions import estimate_refusal_directions
    from ddo_defense.fingerprint import fingerprint, suggest_config
    from ddo_defense.models import ModelAdapter

    adapter = ModelAdapter.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    harmful, harmless = _load_prompts(args.n_probes)

    dirs = estimate_refusal_directions(
        adapter, harmful, harmless, batch_size=args.batch_size
    )
    r_by_layer = [dirs[i] for i in range(adapter.n_layers)]

    fp = fingerprint(
        adapter, harmful, r_by_layer,
        n_harmful=args.n_harmful, batch_size=args.batch_size, seed=args.seed,
    )
    suggestion = suggest_config(fp)

    print("\nfingerprint")
    print(f"  gate architecture      {fp['gate_arch']}")
    zone_start = fp["zone_start"]
    if zone_start >= fp.get("architecture", {}).get("n_layers", zone_start + 1):
        print("  causal zone            none found")
    else:
        print(f"  causal zone starts at  layer {zone_start}")
    print(f"  zone coverage          {fp['zone_coverage']:.2f}")
    print(f"  attack compliance      {fp['rfa_compliance']:.2f}")
    ratio = fp["d0_to_d60_ratio"]
    if ratio != ratio:
        print("  reader cone ratio      not measured (no refusal drop either way)")
    else:
        print(f"  reader cone ratio      {ratio:.2f}")

    print("\nsuggested starting point")
    print(
        f"  layers        {suggestion['layer_start']} to "
        f"{suggestion['layer_end'] - 1} (layer_end={suggestion['layer_end']} is "
        f"exclusive)"
    )
    print(f"  compile mode  {suggestion['compile_mode']}")
    print(f"  because       {suggestion['layer_band_rationale']}")
    for note in suggestion["notes"]:
        print(f"  note          {note}")
    for warning in suggestion["warnings"]:
        print(f"  WARNING       {warning}")

    print(
        "\nThis is a starting range, not a configuration. Pass it to ddo-tune, "
        "which searches around it."
    )

    _write_json({"fingerprint": fp, "suggestion": suggestion}, args.output)
    adapter.free()
    return 0


# ----------------------------------------------------------------------- tune

def tune_main(argv: Optional[Sequence[str]] = None) -> int:
    """Search for a per-model configuration."""
    parser = argparse.ArgumentParser(
        prog="ddo-tune",
        description="Search for a DDO configuration for one model. This is the "
                    "normal way to use the library: configurations do not "
                    "transfer between models.",
    )
    _add_model_args(parser)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--judges", nargs="+", default=["llamaguard2"],
                        help="Judges for scoring trials. One is the default "
                             "because every trial is scored.")
    parser.add_argument("--xstest_floor", type=float, default=0.85,
                        help="Minimum benign compliance a trial must keep")
    parser.add_argument("--dev_size", type=int, default=50,
                        help="Held-out benign prompts for the floor. These are "
                             "excluded from reported scores.")
    parser.add_argument("--n_train", type=int, default=128,
                        help="Probe prompts per class for optimisation and DIM "
                             "estimation (128 harmful + 128 safe).")
    parser.add_argument("--eval_set", default="harmful_val",
                        choices=["harmful_val", "harmful_test",
                                 "jailbreakbench", "harmbench_standard"],
                        help="Prompts trials are scored on. The default is the "
                             "validation split, so selection does not happen on "
                             "a reported benchmark.")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suggestion", default=None,
                        help="JSON from ddo-fingerprint, to seed the layer band")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    from ddo_defense.tuning import tune

    suggestion = None
    if args.suggestion:
        with open(args.suggestion) as fh:
            loaded = json.load(fh)
        suggestion = loaded.get("suggestion", loaded)
        print(f"  seeding the search from {args.suggestion}")

    result = tune(
        args.model_path,
        n_trials=args.n_trials,
        suggestion=suggestion,
        judges=tuple(args.judges),
        xstest_floor=args.xstest_floor,
        dev_size=args.dev_size,
        n_train=args.n_train,
        eval_set=args.eval_set,
        n_eval=args.n_eval,
        seed=args.seed,
        output_path=args.output,
        trust_remote_code=args.trust_remote_code,
    )

    if result["best_params"] is None:
        return 1
    print("\nbest configuration")
    for key, value in result["best_params"].items():
        print(f"  {key:20s} {value}")
    return 0


# ---------------------------------------------------------------------- apply

def build_apply_parser() -> argparse.ArgumentParser:
    """The ``ddo-apply`` parser.

    Separate from :func:`apply_main` so the property every tunable flag has to
    satisfy -- parsing to ``None`` when omitted, so it cannot silently override
    ``--config`` -- can be checked directly instead of by reading the source.
    """
    parser = argparse.ArgumentParser(
        prog="ddo-apply",
        description="Apply a DDO configuration to a model and save it. Use a "
                    "configuration from ddo-tune for this model; one taken from "
                    "another model can leave it less safe than undefended.",
    )
    _add_model_args(parser)
    parser.add_argument("--config", default=None,
                        help="JSON from ddo-tune; its best_params are used")
    parser.add_argument("--layer_start", type=int, default=None)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--init_beta", type=float, default=None)
    parser.add_argument("--init_scale", type=float, default=None)
    parser.add_argument("--confusion_lambda", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    # default=None, not 1: a real default would always override --config.
    parser.add_argument("--n_decoys", type=int, default=None,
                        help="Decoy neurons per layer. One unless a tuned "
                             "config asks for more.")
    parser.add_argument("--n_readers", type=int, default=None,
                        help="Reader groups. 1 is the shared reader; above 1 "
                             "each group reads its own perturbation of r_hat.")
    parser.add_argument("--reader_gamma", type=float, default=None,
                        help="Reader diversity when --n_readers > 1.")
    parser.add_argument("--compile_mode", choices=["replace", "additive"], default=None)
    parser.add_argument("--n_train", type=int, default=128,
                        help="Probe prompts per class for optimisation and DIM "
                             "estimation (128 harmful + 128 safe).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default=None, help="Where to save the model")
    parser.add_argument("--skip_coherence_gate", action="store_true",
                        help="Save even if the model fails the coherence gate. "
                             "Only for inspecting a failure.")
    return parser


def apply_main(argv: Optional[Sequence[str]] = None) -> int:
    """Apply a configuration and save the defended checkpoint."""
    args = build_apply_parser().parse_args(argv)

    config_payload = None
    if args.config:
        with open(args.config) as fh:
            config_payload = json.load(fh)
        print(f"  loaded configuration from {args.config}")

    params = merge_config(
        DEFAULT_LLAMA3_CONFIG,
        config_payload,
        {key: getattr(args, key, None) for key in CONFIG_KEYS},
    )

    if not args.config:
        print(
            "  note: no --config given, so the defaults are the built-in "
            "Llama-3 configuration. On any other model, run ddo-tune first."
        )

    print("  configuration:")
    for key, value in params.items():
        print(f"    {key:20s} {value}")

    from ddo_defense.coherence import coherence_gate
    from ddo_defense.tuning import apply_ddo_from_params

    harmful, harmless = _load_prompts(args.n_train)
    adapter, info = apply_ddo_from_params(
        args.model_path, params,
        harmful_instructions=harmful, harmless_instructions=harmless,
        seed=args.seed, trust_remote_code=args.trust_remote_code, verbose=True,
    )

    report = coherence_gate(adapter.model, adapter.tokenizer)
    print("\n" + report.summary())

    if not report.passed and not args.skip_coherence_gate:
        print(
            "\nNot saving. This checkpoint is degenerate, and its attack-success "
            "and benign-compliance numbers would both be meaningless. Re-tune, "
            "or pass --skip_coherence_gate to save it anyway for inspection."
        )
        adapter.free()
        return 1

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        adapter.model.save_pretrained(args.save_path)
        adapter.tokenizer.save_pretrained(args.save_path)
        info["coherence_passed"] = report.passed
        with open(os.path.join(args.save_path, "ddo_defense.json"), "w") as fh:
            json.dump(info, fh, indent=2, default=str)
        print(f"  saved to {args.save_path}")
    else:
        print("  no --save_path given, so nothing was written")

    adapter.free()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch to a subcommand when run as ``python -m ddo_defense.cli``."""
    commands = {
        "fingerprint": fingerprint_main,
        "tune": tune_main,
        "apply": apply_main,
    }
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in commands:
        print("usage: python -m ddo_defense.cli {fingerprint,tune,apply} ...")
        print("\nTypical order:")
        print("  fingerprint   measure where refusal lives, get a layer band")
        print("  tune          search for a configuration for this model")
        print("  apply         apply a configuration and save the checkpoint")
        return 0 if not argv else 2
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
