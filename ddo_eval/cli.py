"""``ddo-eval``: capability benchmarks, safety judging, and reporting.

Select work with ``--run``.  Judging defaults to all three judges and reports the
mean, with the per-judge numbers kept beside it.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

TASKS = ("mmlu", "mtbench", "xstest", "judge", "aggregate")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ddo-eval",
        description="Evaluate a checkpoint: capability, over-refusal, and "
                    "attack success across three judges.",
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_id", default=None,
                        help="Name for this model in the results directory")
    parser.add_argument("--run", nargs="+", default=["xstest"], choices=list(TASKS) + ["all"])
    parser.add_argument("--output_dir", default="results")
    from ddo_eval.judges import JUDGE_NAMES

    parser.add_argument("--judges", nargs="+", default=None,
                        choices=list(JUDGE_NAMES),
                        help="Defaults to all three: harmbench_cls, llamaguard2, strongreject")
    parser.add_argument("--completions", default=None,
                        help="JSON of completions to judge; expects a 'completions' list")
    parser.add_argument("--dev_size", type=int, default=50,
                        help="XSTest prompts held out for tuning. 0 scores all "
                             "250, which is only comparable when nothing was "
                             "tuned against XSTest.")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="MMLU questions per subject")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args(argv)
    exit_status = 0

    tasks = list(TASKS) if "all" in args.run else list(dict.fromkeys(args.run))
    model_id = args.model_id or (
        os.path.basename(args.model_path.rstrip("/")) if args.model_path else "model"
    )
    model_dir = os.path.join(args.output_dir, model_id)
    # Only created when a task will write into it.  `--run aggregate` with no
    # --model_path falls back to the model_id "model", and an empty directory
    # there is registered by aggregate_results as a model with no results.
    if set(tasks) - {"aggregate"}:
        os.makedirs(model_dir, exist_ok=True)

    needs_model = {"mmlu", "mtbench", "xstest"} & set(tasks)
    if needs_model and not args.model_path:
        parser.error(f"--model_path is required for {sorted(needs_model)}")
    # Checked here rather than inside the judge block: that block runs after the
    # generation benchmarks, so a late exit would discard hours of work.
    if "judge" in tasks and not args.completions:
        parser.error("--completions is required for --run judge")

    if "mmlu" in tasks:
        from ddo_eval.benchmarks.mmlu import run_mmlu

        result = run_mmlu(
            args.model_path, model_dir,
            max_samples=args.max_samples, num_gpus=args.num_gpus,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"  MMLU: {result.get('score')}")

    if "mtbench" in tasks:
        from ddo_eval.benchmarks.mtbench import run_mtbench

        result = run_mtbench(
            args.model_path, model_dir, model_id, num_gpus_total=args.num_gpus
        )
        print(f"  MT-Bench: {result.get('score')}")

    if "xstest" in tasks:
        from ddo_defense.models import ModelAdapter
        from ddo_eval.benchmarks.xstest import run_xstest

        adapter = ModelAdapter.from_pretrained(
            args.model_path, trust_remote_code=args.trust_remote_code
        )
        try:
            result = run_xstest(
                adapter, dev_size=args.dev_size,
                output_path=os.path.join(model_dir, "xstest.json"),
            )
        finally:
            adapter.free()
        score = result.get("score")
        print(
            f"  XSTest compliance: {score * 100:.1f}% over {result['n_reported']} "
            f"prompts ({result['n_dev_held_out']} held out)"
            if score is not None else "  XSTest: no score"
        )

    if "judge" in tasks:
        from ddo_eval.judges import DEFAULT_JUDGES, score_all

        with open(args.completions) as fh:
            payload = json.load(fh)
        completions = payload.get("completions", []) if isinstance(payload, dict) else payload
        attack = payload.get("attack", "attack") if isinstance(payload, dict) else "attack"
        if not isinstance(completions, list) or not completions:
            parser.error("--completions must contain a non-empty list of prompt/response records")

        judges = tuple(args.judges) if args.judges else DEFAULT_JUDGES
        # The classifier judge is a 13B model; without this it would load with
        # tensor_parallel_size=1 however many GPUs were asked for.
        result = score_all(
            completions, judges,
            judge_kwargs={"harmbench_cls": {"num_gpus": args.num_gpus}},
        )
        result["attack"] = attack
        with open(os.path.join(model_dir, f"judging_{attack}.json"), "w") as fh:
            json.dump(result, fh, indent=2, allow_nan=False)

        for name in result["judges_requested"]:
            path = os.path.join(model_dir, f"scores_{name}_attack_{attack}.json")
            with open(path, "w") as fh:
                json.dump({
                    "asr": result["asr"].get(name), "judge": name, "attack": attack,
                    "error": result["errors"].get(name),
                    "n_invalid": result["invalid_judgments"].get(name),
                    "n_completions": result["n_completions"],
                }, fh, indent=2, allow_nan=False)

        mean = result["mean_asr"]
        print(
            f"  mean attack success over {len(result['judges_run'])} judges: "
            f"{mean * 100:.1f}%" if mean is not None
            else "  incomplete judging: aggregate ASR is unavailable"
        )
        if not result["complete"]:
            exit_status = 1

    if "aggregate" in tasks:
        from ddo_eval.aggregate import aggregate_results, format_table, save

        table = aggregate_results(args.output_dir)
        print()
        print(format_table(table))
        paths = save(table, args.output_dir)
        print(f"\n  wrote {paths['json']} and {paths['csv']}")

    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
