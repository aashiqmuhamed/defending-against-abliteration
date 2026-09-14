"""Refusal Feature Ablation: the attack this library defends against.

The attack estimates the refusal direction on whatever model it is given and
projects it out of the forward pass.  It needs no access to the original
weights and no gradient steps, which is what makes it cheap and what makes a
defense that merely hides the direction insufficient.

Two things the caller controls matter, and both are plain parameters here rather
than separate entry points:

``rank``
    How many directions to ablate.  Rank 1 is the published attack.  Higher rank
    removes a subspace, which is strictly stronger.
``n_probes``
    How many prompts the attacker uses to estimate the direction.  A defense
    evaluated only at 128 probes may not hold at 1024.

Ablation surfaces:

``three_point``
    Project at the block input, the attention output and the MLP output.  The
    strongest structural form, since no sublayer can write the direction back.
    This is the default.
``residual_stream``
    Project at the block input only.
``residual_stream_patched``
    Block input, plus adding back the mean harmless projection.

Report several ranks and probe budgets.  A single favourable setting is not
evidence of robustness.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from tqdm import tqdm

from ddo_defense.directions import (
    compute_rank_k_directions,
    filter_probes_by_refusal,
    get_mean_activations,
)
from ddo_defense.hooks import (
    get_activation_addition_input_pre_hook,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)

ABLATION_MODES = ("three_point", "residual_stream", "residual_stream_patched")

#: The four standard reported variants: two surfaces x two evaluation sets.
RFA_VARIANTS: Dict[str, Dict[str, str]] = {
    "RFA": {"ablation_mode": "three_point", "eval_set": "jailbreakbench"},
    "RFA_residual": {"ablation_mode": "residual_stream", "eval_set": "jailbreakbench"},
    "RFA_harmbench": {"ablation_mode": "three_point", "eval_set": "harmbench_standard"},
    "RFA_residual_harmbench": {
        "ablation_mode": "residual_stream", "eval_set": "harmbench_standard",
    },
}


# --------------------------------------------------------------------------
# Direction estimation, from the attacker's side
# --------------------------------------------------------------------------

def compute_attack_directions(
    adapter,
    *,
    rank: int = 1,
    n_probes: int = 128,
    filter_probes: bool = False,
    harmful_instructions: Optional[Sequence[str]] = None,
    harmless_instructions: Optional[Sequence[str]] = None,
    seed: int = 42,
    verbose: bool = True,
    stats: Optional[Dict[str, int]] = None,
) -> Tensor:
    """Estimate the attack subspace on the model as given.

    Draws ``n_probes`` prompts from the train splits, optionally drops prompts
    whose behaviour contradicts their label, and returns
    ``[n_layers, rank, d_model]``.

    ``filter_probes`` keeps harmful prompts the model actually refuses and
    harmless prompts it actually answers.  This sharpens the estimate; turn it
    off to measure the attack without that advantage.

    The prompt pools are finite, so a large ``n_probes`` is capped by what is
    available and a warning says so.  Pass a ``stats`` dict to receive the counts
    actually used: reporting a budget the attack did not really have would
    understate the defense.
    """
    if harmful_instructions is None or harmless_instructions is None:
        from ddo_defense.data import load_dataset_split

        harmful_all = load_dataset_split("harmful", "train", instructions_only=True)
        harmless_all = load_dataset_split("harmless", "train", instructions_only=True)
        rng = random.Random(seed)
        if verbose and n_probes > min(len(harmful_all), len(harmless_all)):
            print(
                f"  note: {n_probes} probes requested but the pools hold "
                f"{len(harmful_all)} harmful and {len(harmless_all)} harmless; "
                f"using what is available"
            )
        harmful_instructions = rng.sample(harmful_all, min(n_probes, len(harmful_all)))
        harmless_instructions = rng.sample(harmless_all, min(n_probes, len(harmless_all)))
    else:
        harmful_instructions = list(harmful_instructions)[:n_probes]
        harmless_instructions = list(harmless_instructions)[:n_probes]

    if verbose:
        print(f"  probes: {len(harmful_instructions)} harmful, "
              f"{len(harmless_instructions)} harmless")

    if filter_probes:
        harmful_instructions, harmless_instructions = filter_probes_by_refusal(
            adapter, harmful_instructions, harmless_instructions,
        )
        if verbose:
            print(f"  after filtering: {len(harmful_instructions)} harmful, "
                  f"{len(harmless_instructions)} harmless")

    if stats is not None:
        stats["n_harmful_used"] = len(harmful_instructions)
        stats["n_harmless_used"] = len(harmless_instructions)
        stats["n_probes_used"] = min(len(harmful_instructions), len(harmless_instructions))

    adapter.model.eval()
    return compute_rank_k_directions(
        adapter, harmful_instructions, harmless_instructions, k=rank
    )


def compute_mean_harmless_projection(
    adapter,
    harmless_instructions: Sequence[str],
    directions: Tensor,
) -> Tensor:
    """Mean harmless component along each layer's direction.

    Only needed by ``residual_stream_patched``, which adds this back after
    projecting, so the activation lands at the harmless mean rather than at zero.

    Computed for the leading direction of each layer only, so with ``rank > 1``
    the remaining directions are projected to zero with nothing added back.  Use
    the patched surface at rank 1.
    """
    means = get_mean_activations(
        adapter.model, list(harmless_instructions),
        adapter.tokenize_instructions_fn, adapter.blocks, positions=(-1,),
    )[0]
    out = torch.zeros_like(means)
    for li in range(means.shape[0]):
        d = directions[li, 0].to(means.device, means.dtype)
        d = d / (d.norm() + 1e-8)
        out[li] = (means[li] @ d) * d
    return out


# --------------------------------------------------------------------------
# Hook construction
# --------------------------------------------------------------------------

def build_ablation_hooks(
    adapter,
    directions: Tensor,
    *,
    ablation_mode: str = "three_point",
    mean_harmless: Optional[Tensor] = None,
):
    """Build ``(fwd_pre_hooks, fwd_hooks)`` implementing the chosen surface.

    ``directions`` is ``[n_layers, rank, d_model]``.  Zero rows are skipped, so a
    padded tensor is safe to pass.
    """
    if ablation_mode not in ABLATION_MODES:
        raise ValueError(
            f"Unknown ablation_mode {ablation_mode!r}. Choose from {ABLATION_MODES}"
        )

    blocks, attns, mlps = adapter.blocks, adapter.attn_modules, adapter.mlp_modules
    n_layers = min(adapter.n_layers, directions.shape[0])
    pre, post = [], []

    for layer in range(n_layers):
        for k_idx in range(directions.shape[1]):
            direction = directions[layer, k_idx]
            if direction.norm() < 1e-10:
                continue
            pre.append((blocks[layer],
                        get_direction_ablation_input_pre_hook(direction=direction)))
            if ablation_mode == "three_point":
                post.append((attns[layer],
                             get_direction_ablation_output_hook(direction=direction)))
                post.append((mlps[layer],
                             get_direction_ablation_output_hook(direction=direction)))

    if ablation_mode == "residual_stream_patched":
        if mean_harmless is None:
            raise ValueError(
                "residual_stream_patched needs mean_harmless; compute it with "
                "compute_mean_harmless_projection"
            )
        for layer in range(n_layers):
            vec = mean_harmless[layer]
            if vec.norm() > 1e-10:
                pre.append((blocks[layer], get_activation_addition_input_pre_hook(
                    vector=vec, coeff=torch.tensor(1.0))))

    return pre, post


# --------------------------------------------------------------------------
# Running the attack
# --------------------------------------------------------------------------

def generate_under_attack(
    adapter,
    dataset: Sequence[Dict],
    directions: Optional[Tensor] = None,
    *,
    ablation_mode: str = "three_point",
    mean_harmless: Optional[Tensor] = None,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    show_progress: bool = True,
) -> List[Dict[str, str]]:
    """Generate completions with the attack applied.

    Pass ``directions=None`` for the undefended, unattacked baseline.  Returns
    ``{"prompt", "response"}`` records ready for a judge.
    """
    pre, post = ((), ())
    if directions is not None:
        pre, post = build_ablation_hooks(
            adapter, directions,
            ablation_mode=ablation_mode, mean_harmless=mean_harmless,
        )

    instructions = [x["instruction"] for x in dataset]
    completions: List[Dict[str, str]] = []

    n_failed = 0
    rng = range(0, len(instructions), batch_size)
    for i in (tqdm(rng, desc="attack gen") if show_progress else rng):
        batch = instructions[i:i + batch_size]
        try:
            responses = adapter.generate(
                batch,
                max_new_tokens=max_new_tokens,
                batch_size=len(batch),
                fwd_pre_hooks=pre,
                fwd_hooks=post,
            )
        except Exception as exc:
            # One bad batch should not lose the whole run, but the records must
            # not be scoreable either way.  A placeholder string is judged like
            # any other response, and a short non-refusing one reads as a
            # SUCCESSFUL ATTACK to every substring scorer, so a generation
            # failure would inflate attack success.  Mark the records instead and
            # let the caller decide.
            print(f"  warning: generation failed for batch at {i}: {exc}")
            completions.extend(
                {"prompt": p, "response": None, "error": str(exc)} for p in batch
            )
            n_failed += len(batch)
            continue

        completions.extend(
            {"prompt": p, "response": r} for p, r in zip(batch, responses)
        )

    if n_failed:
        print(
            f"  warning: {n_failed}/{len(instructions)} prompts have no generation. "
            f"Their records carry response=None and no judge will score them."
        )

    return completions


def run_rfa(
    adapter,
    *,
    rank: int = 1,
    n_probes: int = 128,
    ablation_mode: str = "three_point",
    eval_set: str = "jailbreakbench",
    eval_data: Optional[Sequence[Dict]] = None,
    filter_probes: bool = False,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """Estimate directions, then generate under attack.  One call, one variant.

    Returns the completions plus the settings used, so a result is
    self-describing rather than depending on the caller's memory of the flags.
    """
    if eval_data is None:
        from ddo_defense.data import load_eval_set

        eval_data = load_eval_set(eval_set)

    probe_stats: Dict[str, int] = {}
    directions = compute_attack_directions(
        adapter, rank=rank, n_probes=n_probes,
        filter_probes=filter_probes, seed=seed, verbose=verbose,
        stats=probe_stats,
    )

    mean_harmless = None
    if ablation_mode == "residual_stream_patched":
        from ddo_defense.data import load_dataset_split

        harmless = load_dataset_split("harmless", "train", instructions_only=True)
        mean_harmless = compute_mean_harmless_projection(
            adapter, harmless[:n_probes], directions
        )

    completions = generate_under_attack(
        adapter, eval_data, directions,
        ablation_mode=ablation_mode, mean_harmless=mean_harmless,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
        show_progress=verbose,
    )

    return {
        "attack": "RFA",
        "rank": rank,
        "n_probes_requested": n_probes,
        "n_probes_used": probe_stats.get("n_probes_used", n_probes),
        "n_harmful_probes": probe_stats.get("n_harmful_used"),
        "n_harmless_probes": probe_stats.get("n_harmless_used"),
        "ablation_mode": ablation_mode,
        "eval_set": eval_set,
        "filter_probes": filter_probes,
        "seed": seed,
        "n_completions": len(completions),
        "completions": completions,
    }


def run_rfa_variant(adapter, variant: str, **kwargs) -> Dict[str, object]:
    """Run one of the four named variants in :data:`RFA_VARIANTS`."""
    if variant not in RFA_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Choose from {sorted(RFA_VARIANTS)}"
        )
    settings = dict(RFA_VARIANTS[variant])
    settings.update(kwargs)
    result = run_rfa(adapter, **settings)
    result["attack"] = variant
    return result
