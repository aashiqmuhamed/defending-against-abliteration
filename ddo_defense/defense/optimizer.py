"""Gradient optimization of the decoy directions.

Weight surgery alone writes a random decoy direction.  Optimizing it is what
makes DDO work: the direction is trained so that the defended model keeps
refusing harmful prompts, keeps its benign behaviour, and presents an attacker
re-estimating the refusal direction with a misleading answer.

Three losses, plus one guard:

``refusal``
    Cross-entropy on the model's own refusal completions for harmful prompts, so
    refusal survives the surgery.
``retain``
    KL against the undefended model on harmless prompts, so general behaviour is
    preserved.
``confusion``
    The term that does the defending. With ``self_rfa_kl`` the attack is
    simulated during training: mean-difference directions are estimated on the
    *defended* model, ablated, and the defended and attacked logits are pushed
    together. Invariance to its own best attack is the objective.
``refusal_score``
    A guard on first-token refusal. Without it the confusion term can reach
    invariance the cheap way, by simply never refusing.

Training runs through nnsight so the injected neuron is differentiable, and it
simulates the exact arithmetic that :func:`compile_to_weights` later writes into
the weights: the original neuron's contribution is subtracted and the decoy's
added, so what is trained is what is deployed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ddo_defense.defense.basis import build_decoy_basis
from ddo_defense.defense.surgery import (
    apply_ddo_to_layer,
    build_trigger_set,
    select_neuron_indices,
)
from ddo_defense.mlp import (
    get_activation_fn,
    get_glu_handles,
    get_intermediate_size,
)

#: Layernorm whose output the injector reads as the MLP input, in priority order.
#:
#: The quantity DDO needs is ``x_tilde = RMSNorm(h_attn)``, the tensor the gate
#: and up projections actually read.  On Llama-like blocks that is
#: ``post_attention_layernorm``.  Gemma-2 has both names: there
#: ``post_attention_layernorm`` normalises the attention output before the
#: residual add, and ``pre_feedforward_layernorm`` is the one feeding the MLP,
#: so it has to be preferred wherever it exists.
_MLP_INPUT_NORMS = ("pre_feedforward_layernorm", "post_attention_layernorm")


def _einsum_dot(activation: Tensor, direction: Tensor) -> Tensor:
    """``<direction, activation>`` over the last axis."""
    return (activation * direction).sum(dim=-1)


def _outer(scalars: Tensor, vector: Tensor) -> Tensor:
    """Broadcast ``scalars[..., None] * vector`` back to full width."""
    return scalars.unsqueeze(-1) * vector


def projection_einops(activation: Tensor, direction: Tensor) -> Tensor:
    """Component of ``activation`` along ``direction``."""
    return _outer(_einsum_dot(activation, direction), direction)


def _sha1_of_prompts(prompts: Sequence[str]) -> str:
    """Stable digest of a prompt list, for validating a target cache."""
    h = hashlib.sha1()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def kl_div_fn(logits_a: Tensor, logits_b: Tensor, reduction: str = "batchmean") -> Tensor:
    """KL divergence between two logit tensors, computed in float64."""
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)
    return F.kl_div(
        F.log_softmax(logits_a, dim=-1),
        F.softmax(logits_b, dim=-1),
        reduction=reduction,
    )


def refusal_margin(
    logits: Tensor,
    refusal_toks: Sequence[int],
    compliance_toks: Sequence[int],
) -> Tensor:
    """First-token logit margin between refusal and compliance prefixes.

    The strongest refusal logit minus the strongest compliance logit. Positive
    means the model is poised to refuse.

    This exists because sequence-level losses can be satisfied by opening with a
    compliance token and pivoting mid-sentence, which is not refusal at
    generation time. Anchoring the first token forecloses that.
    """
    logits = logits.to(torch.float64)
    refusal = logits[:, list(refusal_toks)].max(dim=-1).values
    compliance = logits[:, list(compliance_toks)].max(dim=-1).values
    return refusal - compliance


def refusal_metric(logits: Tensor, refusal_toks: Sequence[int], epsilon: float = 1e-8) -> Tensor:
    """Log-odds that the next token begins a refusal."""
    logits = logits.to(torch.float64)
    probs = F.softmax(logits, dim=-1)
    refusal_probs = probs[:, list(refusal_toks)].sum(dim=-1)
    nonrefusal_probs = 1.0 - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)


def original_neuron_contribution(h, gate_row, up_row, down_col, act):
    """What one MLP neuron contributes to the block output.

    ``act(<gate_row, h>) * <up_row, h> * down_col`` for a gated linear unit.
    Replace-mode surgery destroys this, so training must subtract it to match
    the deployed model.
    """
    return _outer(act(_einsum_dot(h, gate_row)) * _einsum_dot(h, up_row), down_col)


def decoy_neuron_contribution(h, trigger, decoy, beta, scale, act):
    """What a hijacked neuron contributes once DDO has rewritten it.

    ``act(beta * <trigger, h>) * <trigger, h> * scale * decoy``. The projection
    appears twice on purpose: once inside the activation as a gate, and once
    outside as the magnitude, so the unit stays quiet until the trigger fires.
    """
    score = _einsum_dot(h, trigger)
    return _outer(act(beta * score) * score, scale * decoy)


def neuron_delta(
    h, *, trigger, decoy, beta, scale, gate_row, up_row, down_col, act,
    compile_mode: str = "replace",
):
    """Net change to the block output from hijacking one neuron.

    This is the single formula that training and deployment must agree on, and it
    depends on the compile mode, because the two modes leave different weights
    behind.

    ``replace`` overwrites the rows, so the neuron computes the decoy alone and
    its original contribution is gone.

    ``additive`` superposes, so the surviving rows are ``gate + beta * trigger``,
    ``up + trigger`` and ``down + scale * decoy``.  The result is not the decoy
    plus the original: the products cross-multiply.  Simulating replacement while
    compiling additively therefore optimises against a model that is never saved.
    """
    if compile_mode not in ("replace", "additive"):
        raise ValueError(
            f"Unknown compile_mode {compile_mode!r}. Use 'replace' or 'additive'."
        )

    original = original_neuron_contribution(h, gate_row, up_row, down_col, act)

    if compile_mode == "replace":
        new = decoy_neuron_contribution(h, trigger, decoy, beta, scale, act)
    else:
        new = original_neuron_contribution(
            h,
            gate_row + beta * trigger,
            up_row + trigger,
            down_col + scale * decoy,
            act,
        )
    return new - original


def _resolve_mlp_norm_name(block) -> str:
    for name in _MLP_INPUT_NORMS:
        if getattr(block, name, None) is not None:
            return name
    raise ValueError(
        "Could not find the layernorm feeding this layer's MLP. Expected one of "
        f"{_MLP_INPUT_NORMS}."
    )


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------

class DecoyInjector:
    """Holds the learnable decoys and applies them inside an nnsight trace.

    Per layer and per decoy neuron there is a learned direction plus two fixed
    scalar gains: a gate sensitivity ``beta`` and an output ``scale``.  Only the
    directions receive gradients; the gains come from the config and are tuned
    by the hyperparameter search.  Shapes are uniform, so a single-decoy run is
    just ``K = 1``.

    The injection replicates the compiled weight edit exactly.  In replace mode,
    for each hijacked neuron it subtracts what that neuron used to contribute and
    adds

        act(beta * <a, h>) * <a, h> * scale * u

    In additive mode the decoy rows are superposed on the original ones instead,
    so the delta is the crossed product of both rather than the term above.

    where ``act`` is the layer's own activation function.  Using the real
    activation matters: training a decoy against SiLU and deploying it into a
    GeGLU model optimizes the wrong function.
    """

    def __init__(
        self,
        module,
        d_model: int,
        target_layers: Sequence[int],
        r_hats: Dict[int, Tensor],
        orig_neuron_weights: Dict[int, Dict[str, List[Tensor]]],
        *,
        init_dirs: Optional[Dict[int, Tensor]] = None,
        init_beta: float = 5.0,
        init_scale: float = 0.1,
        device: Optional[torch.device] = None,
        activation_fns: Optional[Dict[int, Callable]] = None,
        mlp_norm_names: Optional[Dict[int, str]] = None,
        triggers: Optional[Dict[int, Tensor]] = None,
        seed: int = 42,
        compile_mode: str = "replace",
    ):
        self.module = module
        self.target_layers = list(target_layers)
        self.device = device or torch.device("cuda:0")
        self.r_hats = {l: v.to(self.device) for l, v in r_hats.items()}
        self.activation_fns = activation_fns or {}
        self.mlp_norm_names = mlp_norm_names or {}
        # Readers, one column per decoy group.  Absent means every decoy reads
        # r_hat, which is the shared-reader case.  Whatever is here is also what
        # gets written at compile time, so training and deployment cannot
        # disagree about which coordinate a neuron reads.
        self.triggers = (
            {l: t.to(self.device).float() for l, t in triggers.items()}
            if triggers else {}
        )
        if compile_mode not in ("replace", "additive"):
            raise ValueError(
                f"Unknown compile_mode {compile_mode!r}. Use 'replace' or 'additive'."
            )
        self.compile_mode = compile_mode

        def _to_dev(v):
            if isinstance(v, (list, tuple)):
                return [t.to(self.device).float() for t in v]
            return [v.to(self.device).float()]

        self.orig_neuron = {
            l: {k: _to_dev(v) for k, v in ws.items()}
            for l, ws in orig_neuron_weights.items()
        }

        first = self.target_layers[0]
        self.K = len(self.orig_neuron[first]["up"])

        self.fn_vectors: Dict[int, nn.Parameter] = {}
        # beta and scale are hyperparameters, not learned: they are set from the
        # config (Optuna searches them) and held fixed while the decoy
        # directions are optimised.
        self.betas: Dict[int, Tensor] = {}
        self.scales: Dict[int, Tensor] = {}

        for l in self.target_layers:
            inits: List[Tensor] = []
            g = torch.Generator().manual_seed(seed + l)
            for k in range(self.K):
                if init_dirs is not None and l in init_dirs and k == 0:
                    u = init_dirs[l].detach().clone().float().reshape(-1)
                else:
                    u = torch.randn(d_model, generator=g, dtype=torch.float32)
                # Normalise and move to the injector's device here: the warm
                # start comes from the caller's directions while the random
                # columns are built on CPU, and torch.stack cannot mix devices.
                inits.append((u / (u.norm() + 1e-8)).to(self.device))

            self.fn_vectors[l] = nn.Parameter(
                torch.stack(inits).to(self.device), requires_grad=True
            )
            # Spread betas when there are several decoys, so they do not all
            # start with identical gating behaviour.
            betas = (
                torch.full((1,), float(init_beta))
                if self.K == 1
                else torch.linspace(init_beta * 0.5, init_beta * 1.5, self.K)
            )
            self.betas[l] = betas.to(self.device)
            self.scales[l] = torch.full(
                (self.K,), float(init_scale), device=self.device
            )

        self.orthogonalize()
        # beta and scale are fixed hyperparameters, so this is a guard on the
        # supplied config; it belongs here rather than in the training loop.
        self.clamp_params()

    def _mlp_input(self, layer, layer_idx: int) -> Tensor:
        # Resolve rather than defaulting to a name: _MLP_INPUT_NORMS[0] is
        # pre_feedforward_layernorm, which only Gemma-style blocks have, so a
        # missing map entry would raise on every other architecture.
        name = self.mlp_norm_names.get(layer_idx) or _resolve_mlp_norm_name(layer)
        return getattr(layer, name).output.float()

    def _trigger(self, layer_idx: int, k: int) -> Tensor:
        """The direction decoy ``k`` reads, matching what compilation writes."""
        t = self.triggers.get(layer_idx)
        if t is None:
            return self.r_hats[layer_idx]
        return t[:, k % t.shape[1]]

    def _act(self, layer_idx: int) -> Callable:
        """The activation this layer's MLP applies.

        A decoy trained against SiLU and compiled into a GeGLU model such as
        Gemma-2 optimises a different function than the one that runs at
        deployment, so the activation is read from the layer.  ``F.silu`` is the
        fallback for injectors built without an activation map (SwiGLU).
        """
        return self.activation_fns.get(layer_idx, F.silu)

    def inject(self, layer_idx: int) -> None:
        """Add this layer's decoy delta to the MLP output, inside a trace."""
        layer = self.module.layers[layer_idx]
        orig = self.orig_neuron[layer_idx]
        act = self._act(layer_idx)

        h = self._mlp_input(layer, layer_idx)

        delta = None
        for k in range(self.K):
            term = neuron_delta(
                h,
                trigger=self._trigger(layer_idx, k),
                decoy=self.fn_vectors[layer_idx][k],
                beta=self.betas[layer_idx][k],
                scale=self.scales[layer_idx][k],
                gate_row=orig["gate"][k],
                up_row=orig["up"][k],
                down_col=orig["down"][k],
                act=act,
                compile_mode=self.compile_mode,
            )
            delta = term if delta is None else delta + term

        layer.mlp.output = layer.mlp.output + delta.to(layer.mlp.output.dtype)

    def inject_all(self) -> None:
        for l in self.target_layers:
            self.inject(l)

    def parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for l in self.target_layers:
            params.append(self.fn_vectors[l])
        return params

    def orthogonalize(self) -> None:
        """Keep every decoy orthogonal to ``r`` and to the other decoys.

        Orthogonality to ``r`` is what makes a decoy a decoy: a component along
        the refusal direction would strengthen or weaken refusal instead of
        misdirecting the estimate.  Gram-Schmidt across decoys stops several
        neurons from collapsing onto the same direction.
        """
        with torch.no_grad():
            for l in self.target_layers:
                r = self.r_hats[l].to(self.fn_vectors[l].device)
                U = self.fn_vectors[l].data
                for k in range(self.K):
                    u = U[k] - (U[k] @ r) * r
                    for j in range(k):
                        u = u - (u @ U[j]) * U[j]
                    U[k] = u / (u.norm() + 1e-8)

    def clamp_params(
        self,
        min_beta: float = 0.5,
        max_beta: float = 20.0,
        min_scale: float = 0.01,
        max_scale: float = 5.0,
    ) -> None:
        """Keep beta and scale in a range where the gate stays well-behaved.

        These are fixed hyperparameters, so this is a guard on the supplied
        config rather than a correction applied during training.
        """
        with torch.no_grad():
            for l in self.target_layers:
                self.betas[l].data.clamp_(min_beta, max_beta)
                self.scales[l].data.clamp_(min_scale, max_scale)


# ---------------------------------------------------------------------------
# Dataset plumbing
# ---------------------------------------------------------------------------

class DDODataset(Dataset):
    """Harmful prompts with the refusal and retain targets built from them."""

    def __init__(
        self,
        harmful_prompts: List[str],
        refusal_prompts: List[str],
        refusal_labels: List[Tensor],
        retain_prompts: List[str],
    ):
        self.harmful_prompts = harmful_prompts
        self.refusal_prompts = refusal_prompts
        self.refusal_labels = refusal_labels
        self.retain_prompts = retain_prompts

    def __len__(self) -> int:
        return len(self.harmful_prompts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "harmful_prompt": self.harmful_prompts[idx],
            "refusal_prompt": self.refusal_prompts[idx],
            "refusal_labels": self.refusal_labels[idx],
            "retain_prompt": self.retain_prompts[idx],
        }


def ddo_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Collate a batch, left-padding labels to match left-padded inputs."""
    labels = [item["refusal_labels"] for item in batch]
    max_len = max(l.size(0) for l in labels)
    padded = torch.stack([F.pad(l, (max_len - l.size(0), 0), value=-100) for l in labels])
    return {
        "harmful_prompt": [item["harmful_prompt"] for item in batch],
        "refusal_prompt": [item["refusal_prompt"] for item in batch],
        "refusal_labels": padded,
        "retain_prompt": [item["retain_prompt"] for item in batch],
    }


def generate_completions_nnsight(
    model, prompts: Sequence[str], *, max_new_tokens: int = 30, batch_size: int = 16
) -> List[str]:
    """Greedy-decode with nnsight, deriving the prompt length from the tokenizer.

    Left padding is forced so every row's generation begins at the same index.
    """
    from tqdm import tqdm

    out: List[str] = []
    orig_side = model.tokenizer.padding_side
    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="targets"):
            batch = list(prompts[i:i + batch_size])
            tok_out = model.tokenizer(
                batch, padding=True, add_special_tokens=True,
                return_tensors="pt", truncation=False,
            )
            padded_len = tok_out["input_ids"].shape[1]

            with model.generate(batch, max_new_tokens=max_new_tokens, do_sample=False):
                tokens = model.generator.output.save()

            for j in range(len(batch)):
                out.append(
                    model.tokenizer.decode(
                        tokens.value[j, padded_len:], skip_special_tokens=True
                    )
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        model.tokenizer.padding_side = orig_side

    return out


def build_prompts_and_labels(
    tokenizer,
    harmful_prompts: Sequence[str],
    harmless_prompts: Sequence[str],
    refusal_targets: Sequence[str],
    retain_targets: Sequence[str],
) -> Tuple[List[str], List[Tensor], List[str]]:
    """Build ``(prompt+target, labels)`` pairs with the instruction masked out.

    The instruction and the target are tokenized separately and concatenated, so
    no byte-pair merge straddles the boundary and shifts every label by one.
    """
    refusal_out: List[str] = []
    labels_out: List[Tensor] = []
    retain_out: List[str] = []

    for harmful, harmless, ref_target, ret_target in zip(
        harmful_prompts, harmless_prompts, refusal_targets, retain_targets
    ):
        inst_ids = tokenizer.encode(harmful, add_special_tokens=False)
        target_ids = tokenizer.encode(ref_target, add_special_tokens=False)
        refusal_text = tokenizer.decode(inst_ids + target_ids, skip_special_tokens=False)
        refusal_out.append(refusal_text)

        full_ids = tokenizer.encode(refusal_text, add_special_tokens=True)
        label = torch.tensor(full_ids[1:], dtype=torch.long)
        prompt_ids = tokenizer.encode(harmful, add_special_tokens=True)
        label[: len(prompt_ids) - 1] = -100
        labels_out.append(label)

        retain_out.append(harmless + ret_target)

    return refusal_out, labels_out, retain_out


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class DDOOptimizer:
    """Fits decoy directions, then compiles them into weights.

    Parameters
    ----------
    nn_model
        An ``nnsight.LanguageModel``, needed for differentiable interventions.
    r_by_layer
        Per-layer refusal directions.
    target_layers
        Layers to defend.
    n_decoys
        Neurons hijacked per layer.  Each gets its own learned direction; the
        gate sensitivity and output scale are fixed hyperparameters taken from
        ``init_beta`` and ``init_scale``.
    format_fn
        Maps a raw instruction to a formatted prompt.  Pass
        ``adapter.format_instruction`` so training, direction estimation and
        attacks all use one formatting path.  Falling back to the tokenizer's
        own chat template is fine; hard-coding a family template is not.
    refusal_toks
        Token ids that begin a refusal.  Taken from the adapter when omitted.
    """

    def __init__(
        self,
        nn_model,
        r_by_layer: Sequence[Tensor],
        target_layers: Sequence[int],
        *,
        n_decoys: int = 1,
        init_beta: float = 5.0,
        init_scale: float = 0.1,
        seed: int = 42,
        n_triggers: int = 1,
        trigger_source: str = "single",
        reader_gamma: float = 0.3,
        neuron_selection: str = "low_norm",
        format_fn: Optional[Callable[[str], str]] = None,
        refusal_toks: Optional[Sequence[int]] = None,
        anchor_refusal_toks: Optional[Sequence[int]] = None,
        anchor_compliance_toks: Optional[Sequence[int]] = None,
        compile_mode: str = "replace",
    ):
        self.nn_model = nn_model
        self.target_layers = sorted(target_layers)

        # A layer where harmful and harmless activations coincide has no
        # direction to build a decoy around, and injecting there would write a
        # zero trigger and kill the neuron for nothing.
        degenerate = [
            l for l in self.target_layers if r_by_layer[l].float().norm() < 1e-8
        ]
        if degenerate:
            print(
                f"  skipping {len(degenerate)} layer(s) with a degenerate "
                f"refusal direction: {degenerate}"
            )
            self.target_layers = [
                l for l in self.target_layers if l not in degenerate
            ]
        if not self.target_layers:
            raise ValueError(
                "Every requested layer has a degenerate refusal direction. "
                "These layers provide no nonzero trigger direction. "
                "Check the prompts and the activation hookpoint."
            )
        self.skipped_layers = degenerate
        self.n_decoys = n_decoys
        self.seed = seed
        self.n_triggers = n_triggers
        self.trigger_source = trigger_source
        self.reader_gamma = reader_gamma
        self.neuron_selection = neuron_selection
        # Training must simulate the mode that will be compiled, so the mode is
        # fixed at construction rather than chosen at compile time.
        self.compile_mode = compile_mode

        if n_decoys < 1:
            raise ValueError(
                f"n_decoys={n_decoys} leaves no decoy neurons. At least one per "
                f"layer is required, since the decoy is what misdirects the attack."
            )

        hf_model = nn_model._model
        d_model = int(nn_model.config.hidden_size)
        device = self._resolve_device(nn_model)

        self.tokenizer = nn_model.tokenizer
        self.format_fn = format_fn or self._default_format_fn
        self.refusal_toks = list(refusal_toks) if refusal_toks else self._default_refusal_toks()
        self.anchor_refusal_toks = (
            list(anchor_refusal_toks) if anchor_refusal_toks
            else self._default_anchor_toks(("I", "Sorry", "cannot"))
        )
        self.anchor_compliance_toks = (
            list(anchor_compliance_toks) if anchor_compliance_toks
            else self._default_anchor_toks(("Sure", "Here"))
        )

        r_hats: Dict[int, Tensor] = {}
        init_dirs: Dict[int, Tensor] = {}
        orig_neuron_weights: Dict[int, Dict[str, List[Tensor]]] = {}
        activation_fns: Dict[int, Callable] = {}
        mlp_norm_names: Dict[int, str] = {}
        triggers: Dict[int, Tensor] = {}
        self.layer_compile_data: Dict[int, Dict[str, object]] = {}

        for l in self.target_layers:
            r_hat = F.normalize(r_by_layer[l].float(), dim=0)
            r_hats[l] = r_hat

            U = build_decoy_basis(r_hat, k=2, seed=seed + l, dtype=torch.float32)
            init_dirs[l] = U[:, 1]

            block = hf_model.model.layers[l]
            activation_fns[l] = get_activation_fn(block, hf_model.config)
            mlp_norm_names[l] = _resolve_mlp_norm_name(block)

            up, gate, down = get_glu_handles(block)
            indices = select_neuron_indices(
                n_neurons=n_decoys,
                intermediate_size=get_intermediate_size(block),
                method=neuron_selection,
                down_proj_weight=down.weight.data if neuron_selection == "low_norm" else None,
                seed=seed + l,
            )

            orig_neuron_weights[l] = {
                "up": [up.weight.data[i].detach().clone().float() for i in indices],
                "gate": [gate.weight.data[i].detach().clone().float() for i in indices],
                "down": [down.weight.data[:, i].detach().clone().float() for i in indices],
            }

            # Built once here, used both by the injector during training and by
            # compile_to_weights afterwards.
            if n_triggers > 1:
                triggers[l] = build_trigger_set(
                    r_hat, n_triggers=n_triggers,
                    trigger_source=trigger_source, seed=seed + l,
                    gamma=reader_gamma,
                )

            self.layer_compile_data[l] = {
                "r_hat": r_hat,
                "neuron_indices": list(indices),
            }

        self.injector = DecoyInjector(
            module=nn_model.model,
            d_model=d_model,
            target_layers=self.target_layers,
            r_hats=r_hats,
            orig_neuron_weights=orig_neuron_weights,
            init_dirs=init_dirs,
            init_beta=init_beta,
            init_scale=init_scale,
            device=device,
            activation_fns=activation_fns,
            mlp_norm_names=mlp_norm_names,
            triggers=triggers,
            seed=seed,
            compile_mode=compile_mode,
        )

        print(
            f"DDO: {len(self.target_layers)} layers, {n_decoys} decoy(s) per layer, "
            f"refusal_toks={self.refusal_toks}, device={device}"
        )

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _resolve_device(nn_model) -> torch.device:
        """Single-GPU is required: params and activations must share a device."""
        devices = {p.device for p in nn_model._model.parameters()}
        real = {d for d in devices if d.type != "meta"}
        if len(real) > 1:
            raise RuntimeError(
                f"DDO training requires a single GPU but the model spans {real}. "
                f"Set CUDA_VISIBLE_DEVICES to one device."
            )
        if real:
            return next(iter(real))
        # nnsight can report meta devices while computing on cuda:0.
        return torch.device("cuda:0")

    def _default_format_fn(self, instruction: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"This checkpoint's tokenizer cannot render a chat template "
                f"({type(exc).__name__}: {exc}). Falling back to the raw "
                f"instruction would train the decoy on unformatted prompts while "
                f"the directions were measured on formatted ones. Pass format_fn "
                f"explicitly if the checkpoint has no template."
            ) from exc

    def _default_anchor_toks(self, words) -> List[int]:
        ids: List[int] = []
        for word in words:
            enc = self.tokenizer.encode(word, add_special_tokens=False)
            if enc and int(enc[0]) not in ids:
                ids.append(int(enc[0]))
        return ids

    def _default_refusal_toks(self) -> List[int]:
        enc = self.tokenizer.encode("I", add_special_tokens=False)
        return [int(enc[0])] if enc else []

    def _generate_targets(
        self,
        harmful_instructions: Sequence[str],
        harmless_instructions: Sequence[str],
        *,
        num_target_tokens: int = 30,
        cache_dir: Optional[str] = None,
    ):
        """Format prompts and generate the model's own targets, caching them.

        The cache is validated against the prompts it was written for.  Targets
        are paired with prompts positionally, so a cache belonging to a different
        checkpoint or to a reordered probe set would train the refusal loss
        against continuations that belong to other prompts.
        """
        harmful_prompts = [self.format_fn(i) for i in harmful_instructions]
        harmless_prompts = [self.format_fn(i) for i in harmless_instructions]

        fingerprint = {
            "model": str(getattr(self.nn_model, "name_or_path", "") or ""),
            "n_harmful": len(harmful_prompts),
            "n_harmless": len(harmless_prompts),
            "num_target_tokens": num_target_tokens,
            "prompts_sha1": _sha1_of_prompts(harmful_prompts + harmless_prompts),
        }

        refusal_targets = retain_targets = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            ref_path = os.path.join(cache_dir, "refusal_targets.json")
            ret_path = os.path.join(cache_dir, "retain_targets.json")
            man_path = os.path.join(cache_dir, "targets_manifest.json")
            if all(os.path.exists(x) for x in (ref_path, ret_path, man_path)):
                try:
                    cached = json.load(open(man_path))
                except Exception:
                    cached = None
                if cached == fingerprint:
                    refusal_targets = json.load(open(ref_path))
                    retain_targets = json.load(open(ret_path))
                else:
                    print(
                        "  note: ignoring the cached targets in "
                        f"{cache_dir}; they were written for different prompts "
                        "or a different checkpoint"
                    )

        if refusal_targets is None:
            refusal_targets = generate_completions_nnsight(
                self.nn_model, harmful_prompts, max_new_tokens=num_target_tokens
            )
            self._warn_if_targets_are_not_refusals(refusal_targets)
            if cache_dir:
                json.dump(refusal_targets, open(os.path.join(cache_dir, "refusal_targets.json"), "w"))

        if retain_targets is None:
            retain_targets = generate_completions_nnsight(
                self.nn_model, harmless_prompts, max_new_tokens=num_target_tokens
            )
            if cache_dir:
                json.dump(retain_targets, open(os.path.join(cache_dir, "retain_targets.json"), "w"))

        if len(refusal_targets) != len(harmful_prompts):
            raise RuntimeError(
                f"{len(refusal_targets)} refusal targets for "
                f"{len(harmful_prompts)} harmful prompts. They are paired by "
                f"position, so a mismatch would train against the wrong targets."
            )
        if len(retain_targets) != len(harmless_prompts):
            raise RuntimeError(
                f"{len(retain_targets)} retain targets for "
                f"{len(harmless_prompts)} harmless prompts."
            )

        if cache_dir:
            with open(os.path.join(cache_dir, "targets_manifest.json"), "w") as fh:
                json.dump(fingerprint, fh, indent=2)

        return harmful_prompts, harmless_prompts, refusal_targets, retain_targets

    @staticmethod
    def _warn_if_targets_are_not_refusals(targets: Sequence[str]) -> float:
        """Report how many refusal targets do not look like refusals.

        The targets are the base model's own continuations on the harmful
        probes, which is what makes the refusal term a preservation objective.
        That only holds while the base model actually refuses: any probe it
        complies with contributes a compliance target, so the term would train
        against its own purpose.  Reported rather than corrected, because the
        fix is a better probe set, not a rewritten target.
        """
        from ddo_defense.judges import is_refusal

        if not targets:
            return 0.0
        n_bad = sum(1 for t in targets if not is_refusal(t))
        share = n_bad / len(targets)
        if n_bad:
            print(
                f"  warning: {n_bad}/{len(targets)} refusal targets do not read "
                f"as refusals ({share:.0%}). The refusal loss trains toward the "
                f"base model's own continuation, so those prompts pull toward "
                f"compliance."
            )
        return share

    # -------------------------------------------------------------------- fit

    def fit(
        self,
        harmful_instructions: Sequence[str],
        harmless_instructions: Sequence[str],
        *,
        epochs: int = 1,
        lr: float = 1e-2,
        batch_size: int = 1,
        effective_batch_size: int = 16,
        num_target_tokens: int = 30,
        refusal_lambda: float = 1.0,
        refusal_score_lambda: float = 1.0,
        refusal_score_target: float = 0.0,
        retain_lambda: float = 1.0,
        confusion_lambda: float = 0.5,
        confusion_batch_size: int = 16,
        confusion_objective: str = "self_rfa_kl",
        confusion_n_train: int = 128,
        confusion_update_steps: int = 10,
        confusion_filter_train: bool = True,
        confusion_ablation_mode: str = "three_point",
        patience: int = 5,
        n_lr_reduce: int = 2,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, object]:
        """Run the optimization loop and return the loss history."""
        if confusion_objective not in ("self_rfa_kl",):
            raise ValueError(
                f"Unknown confusion_objective {confusion_objective!r}. "
                "Use 'self_rfa_kl', the differentiable self-attack simulator."
            )
        if confusion_ablation_mode not in ("residual_stream", "three_point"):
            raise ValueError(
                f"Unknown confusion_ablation_mode {confusion_ablation_mode!r}. "
                "Use 'three_point' or 'residual_stream'."
            )

        n = min(len(harmful_instructions), len(harmless_instructions))
        harmful_instructions = list(harmful_instructions)[:n]
        harmless_instructions = list(harmless_instructions)[:n]

        harmful_prompts, harmless_prompts, refusal_targets, retain_targets = \
            self._generate_targets(
                harmful_instructions, harmless_instructions,
                num_target_tokens=num_target_tokens, cache_dir=cache_dir,
            )

        model = self.nn_model
        injector = self.injector
        device = injector.device
        tokenizer = model.tokenizer

        refusal_prompts, refusal_labels, retain_prompts = build_prompts_and_labels(
            tokenizer, harmful_prompts, harmless_prompts, refusal_targets, retain_targets
        )
        dataset = DDODataset(
            harmful_prompts, refusal_prompts, refusal_labels, retain_prompts,
        )
        # Seeded so that a given seed reproduces the batch order, and with it
        # the decoys.  An unseeded shuffle makes two runs of the same config
        # disagree.
        loader_gen = torch.Generator().manual_seed(self.seed)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, collate_fn=ddo_collate, generator=loader_gen,
        )

        optimizer = torch.optim.AdamW(
            injector.parameters(), lr=lr,
            betas=(0.9, 0.98), weight_decay=0.0, amsgrad=True,
        )
        accumulation_steps = max(effective_batch_size // batch_size, 1)

        # Last-token positions must line up with the last real token, so left
        # padding is forced for the duration of training.
        orig_padding_side = tokenizer.padding_side
        orig_pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        history: Dict[str, object] = {
            "train_losses": [], "refusal_losses": [], "refusal_score_losses": [],
            "retain_losses": [], "confusion_losses": [], "refusal_logodds": [],
        }

        import random as _random

        rng = _random.Random(self.seed)
        conf_harmful_pool = list(harmful_prompts)
        conf_harmless_pool = list(harmless_prompts)
        rng.shuffle(conf_harmful_pool)
        rng.shuffle(conf_harmless_pool)
        B_conf = min(confusion_batch_size, len(conf_harmful_pool), len(conf_harmless_pool))

        def _take_window(pool: List[str], start: int, n_take: int) -> List[str]:
            if not pool:
                return []
            # One cursor walks pools of different lengths -- the filtered harmful
            # and harmless sets rarely match -- so reduce it into this pool
            # before slicing.  Otherwise an out-of-range start silently yields a
            # short or empty window instead of ``n_take`` prompts.
            start %= len(pool)
            if n_take >= len(pool):
                return list(pool)
            end = start + n_take
            if end <= len(pool):
                return pool[start:end]
            return pool[start:] + pool[: (end % len(pool))]

        def _score_prompts(prompts: List[str], *, with_injection: bool) -> Tensor:
            if not prompts:
                return torch.empty((0,), dtype=torch.float32)
            scores = []
            with torch.no_grad():
                for i in range(0, len(prompts), max(B_conf, 1)):
                    batch = prompts[i:i + max(B_conf, 1)]
                    with model.trace() as tracer:
                        with tracer.invoke(batch):
                            if with_injection:
                                injector.inject_all()
                            last = model.lm_head.output[:, -1]
                            saved = refusal_metric(last, self.refusal_toks).detach().save()
                    scores.append(saved.value.detach().float().cpu())
            return torch.cat(scores, dim=0)

        if confusion_filter_train and confusion_objective == "self_rfa_kl":
            if verbose:
                print("  filtering confusion pools by refusal score on the defended model")
            harm_scores = _score_prompts(conf_harmful_pool, with_injection=True)
            safe_scores = _score_prompts(conf_harmless_pool, with_injection=True)
            conf_harmful_pool = [
                p for p, s in zip(conf_harmful_pool, harm_scores.tolist()) if s > 0
            ]
            conf_harmless_pool = [
                p for p, s in zip(conf_harmless_pool, safe_scores.tolist()) if s < 0
            ]
            if len(conf_harmful_pool) < 8 or len(conf_harmless_pool) < 8:
                # Early in training the defended model can look broken enough to
                # filter almost everything away; fall back to the clean model.
                if verbose:
                    print("  too few prompts survived; filtering on the clean model instead")
                conf_harmful_pool = list(harmful_prompts)
                conf_harmless_pool = list(harmless_prompts)
                rng.shuffle(conf_harmful_pool)
                rng.shuffle(conf_harmless_pool)
                harm_scores = _score_prompts(conf_harmful_pool, with_injection=False)
                safe_scores = _score_prompts(conf_harmless_pool, with_injection=False)
                conf_harmful_pool = [
                    p for p, s in zip(conf_harmful_pool, harm_scores.tolist()) if s > 0
                ]
                conf_harmless_pool = [
                    p for p, s in zip(conf_harmless_pool, safe_scores.tolist()) if s < 0
                ]
            B_conf = min(B_conf, len(conf_harmful_pool), len(conf_harmless_pool))
            if B_conf == 0:
                raise RuntimeError(
                    "Confusion pools are empty after filtering. Re-run with "
                    "confusion_filter_train=False."
                )
            if verbose:
                print(f"  confusion pools: {len(conf_harmful_pool)} harmful, "
                      f"{len(conf_harmless_pool)} harmless, batch {B_conf}")

        model_dtype = next(model.parameters()).dtype

        def _estimate_mean_input_acts(prompts: List[str]) -> Dict[int, Tensor]:
            sums = {
                l: torch.zeros((model.config.hidden_size,), device=device, dtype=torch.float64)
                for l in self.target_layers
            }
            total = 0
            with torch.no_grad():
                for i in range(0, len(prompts), max(B_conf, 1)):
                    batch = prompts[i:i + max(B_conf, 1)]
                    with model.trace() as tracer:
                        with tracer.invoke(batch):
                            injector.inject_all()
                            saved = {
                                l: model.model.layers[l].input[:, -1].float().sum(0).detach().save()
                                for l in self.target_layers
                            }
                    for l in self.target_layers:
                        sums[l] += saved[l].value.to(device=device, dtype=torch.float64)
                    total += len(batch)
            return {
                l: (sums[l] / max(total, 1)).to(dtype=torch.float32)
                for l in self.target_layers
            }

        # The simulated attack estimates directions on the defended layers only,
        # which is where the decoy lives.  A real attacker estimates at every
        # layer, so this is a cheaper proxy, not the same measurement.
        def _compute_rfa_dirs(harm: List[str], safe: List[str]) -> Dict[int, Tensor]:
            mean_h = _estimate_mean_input_acts(harm)
            mean_s = _estimate_mean_input_acts(safe)
            dirs = {}
            for l in self.target_layers:
                diff = (mean_h[l] - mean_s[l]).float()
                nrm = diff.norm()
                if torch.isfinite(nrm) and nrm > 1e-8:
                    dirs[l] = diff / (nrm + 1e-8)
            return dirs

        def _apply_rfa_ablation(dirs_by_layer: Dict[int, Tensor]) -> None:
            for l, d in dirs_by_layer.items():
                layer = model.model.layers[l]
                d = d.to(device=device, dtype=model_dtype)
                layer.input[:] = layer.input[:] - projection_einops(layer.input[:], d)
                if confusion_ablation_mode == "three_point":
                    layer.self_attn.output[0][:] = (
                        layer.self_attn.output[0][:]
                        - projection_einops(layer.self_attn.output[0][:], d)
                    )
                    layer.mlp.output[:] = (
                        layer.mlp.output[:] - projection_einops(layer.mlp.output[:], d)
                    )

        step = 0
        opt_step = 0
        stopped = False
        lowest_loss = float("inf")
        patience_counter = 0
        lr_reduce_counter = 0
        conf_buf_idx = conf_dir_idx = conf_refusal_idx = 0
        cached_dirs: Optional[Dict[int, Tensor]] = None
        acc = {"refusal": 0.0, "refusal_score": 0.0, "retain": 0.0, "confusion": 0.0}
        bypass_scores: List[float] = []
        n_conf_skipped = 0
        t0 = time.time()

        try:
            for epoch in range(epochs):
                if stopped:
                    break
                if verbose:
                    print(f"\nepoch {epoch}")

                for batch in dataloader:
                    if refusal_lambda > 0:
                        n_label_toks = batch["refusal_labels"].size(1)
                        with model.trace() as tracer:
                            with tracer.invoke(batch["refusal_prompt"]):
                                injector.inject_all()
                                logits = model.lm_head.output[:, -(n_label_toks + 1):-1]
                                ref_loss = F.cross_entropy(
                                    logits.reshape(-1, logits.shape[-1]),
                                    batch["refusal_labels"].to(device).reshape(-1),
                                    ignore_index=-100,
                                )
                                log_ref = ref_loss.detach().item().save()
                            (refusal_lambda * ref_loss).backward()
                        acc["refusal"] += log_ref.value

                    if retain_lambda > 0:
                        with model.trace() as tracer:
                            with tracer.invoke(batch["retain_prompt"]):
                                baseline_logits = model.lm_head.output[:, -num_target_tokens:]
                            with tracer.invoke(batch["retain_prompt"]):
                                injector.inject_all()
                                defended_logits = model.lm_head.output[:, -num_target_tokens:]
                                # Flattened so ``batchmean`` divides by tokens, not
                                # by batch alone.  Without this the retain term is a
                                # SUM over the target positions while the confusion
                                # term is per token, so lambda_ret = lambda_conf = 1
                                # would weight retain by the number of positions.
                                ret_loss = kl_div_fn(
                                    baseline_logits.reshape(-1, baseline_logits.shape[-1]),
                                    defended_logits.reshape(-1, defended_logits.shape[-1]),
                                )
                                log_ret = ret_loss.detach().item().save()
                            (retain_lambda * ret_loss).backward()
                        acc["retain"] += log_ret.value

                    with torch.no_grad():
                        with model.trace() as tracer:
                            with tracer.invoke(batch["harmful_prompt"]):
                                injector.inject_all()
                                last = model.lm_head.output[:, -1]
                                bp = refusal_metric(
                                    last, self.refusal_toks
                                ).detach().mean().item().save()
                        bypass_scores.append(bp.value)

                    step += 1
                    if step % accumulation_steps:
                        continue
                    opt_step += 1

                    if refusal_score_lambda > 0 and conf_harmful_pool:
                        harm_rs = _take_window(conf_harmful_pool, conf_refusal_idx, B_conf)
                        conf_refusal_idx = (conf_refusal_idx + B_conf) % max(len(conf_harmful_pool), 1)
                        with model.trace() as tracer:
                            with tracer.invoke(harm_rs):
                                injector.inject_all()
                                last = model.lm_head.output[:, -1]
                                margin = refusal_margin(
                                    last,
                                    self.anchor_refusal_toks,
                                    self.anchor_compliance_toks,
                                )
                                # Hinge on the first-token margin: once refusal
                                # tokens lead compliance tokens by the target, this
                                # term and its gradient are zero.
                                rs_loss = torch.clamp(
                                    refusal_score_target - margin, min=0.0
                                ).mean()
                                log_rs = rs_loss.detach().item().save()
                            (refusal_score_lambda * accumulation_steps * rs_loss).backward()
                        acc["refusal_score"] = log_rs.value

                    if confusion_lambda > 0:
                        n_train = min(
                            confusion_n_train, len(conf_harmful_pool), len(conf_harmless_pool)
                        )
                        harm_dir = _take_window(conf_harmful_pool, conf_dir_idx, n_train)
                        safe_dir = _take_window(conf_harmless_pool, conf_dir_idx, n_train)
                        conf_dir_idx = (conf_dir_idx + n_train) % max(len(conf_harmful_pool), 1)

                        # Re-estimating the attack every step would dominate the
                        # cost, so the directions are refreshed periodically.
                        if cached_dirs is None or opt_step % max(confusion_update_steps, 1) == 0:
                            if verbose:
                                print(f"    re-estimating attack directions on {n_train} prompts")
                            cached_dirs = _compute_rfa_dirs(harm_dir, safe_dir)

                        harm_eval = _take_window(conf_harmful_pool, conf_buf_idx, B_conf)
                        safe_eval = _take_window(conf_harmless_pool, conf_buf_idx, B_conf)
                        conf_buf_idx = (conf_buf_idx + B_conf) % max(len(conf_harmful_pool), 1)

                        if not cached_dirs:
                            # Every target layer's mean difference was degenerate,
                            # so there is no attack to be invariant to and the term
                            # was not evaluated.  Recording 0.0 would report the
                            # best possible confusion objective -- perfectly
                            # invariant to its own attack -- for a step that never
                            # ran, and that value feeds train_loss and therefore
                            # early stopping.
                            n_conf_skipped += 1
                            print(
                                "  warning: attack directions degenerate on all "
                                "target layers; confusion term skipped this step"
                            )
                            acc["confusion"] = float("nan")
                            cached_dirs = None  # force a re-estimate next step
                        else:
                            with model.trace() as tracer:
                                with tracer.invoke(harm_eval):
                                    injector.inject_all()
                                    defended_h = model.lm_head.output[:, -1]
                                with tracer.invoke(safe_eval):
                                    injector.inject_all()
                                    defended_s = model.lm_head.output[:, -1]
                                with tracer.invoke(harm_eval):
                                    injector.inject_all()
                                    _apply_rfa_ablation(cached_dirs)
                                    attacked_h = model.lm_head.output[:, -1]
                                with tracer.invoke(safe_eval):
                                    injector.inject_all()
                                    _apply_rfa_ablation(cached_dirs)
                                    attacked_s = model.lm_head.output[:, -1]

                                conf_loss = (
                                    kl_div_fn(defended_h, attacked_h).mean()
                                    + kl_div_fn(defended_s, attacked_s).mean()
                                )
                                log_conf = conf_loss.detach().item().save()
                                (confusion_lambda * accumulation_steps * conf_loss).backward()
                            acc["confusion"] = log_conf.value

                    # Keep the gradient tangent to the unit sphere; the radial part
                    # would only change a norm that orthogonalize() resets anyway.
                    with torch.no_grad():
                        for l in self.target_layers:
                            v = injector.fn_vectors[l]
                            if v.grad is not None:
                                for k in range(injector.K):
                                    v.grad[k].sub_(projection_einops(v.grad[k], v.data[k]))

                    for p in injector.parameters():
                        if p.grad is not None:
                            p.grad.div_(accumulation_steps)

                    torch.nn.utils.clip_grad_norm_(injector.parameters(), 10.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    injector.orthogonalize()

                    acc["refusal"] /= accumulation_steps
                    acc["retain"] /= accumulation_steps

                    # A skipped confusion term is absent, not zero.  It is left out of
                    # the composite so train_loss describes what was actually
                    # optimised, and recorded as None so the history does not read as
                    # a perfect confusion objective on that step.
                    conf_evaluated = math.isfinite(acc["confusion"])
                    train_loss = (
                        refusal_lambda * acc["refusal"]
                        + refusal_score_lambda * acc["refusal_score"]
                        + retain_lambda * acc["retain"]
                        + (confusion_lambda * acc["confusion"] if conf_evaluated else 0.0)
                    )

                    history["train_losses"].append(train_loss)
                    history["refusal_losses"].append(acc["refusal"])
                    history["refusal_score_losses"].append(acc["refusal_score"])
                    history["retain_losses"].append(acc["retain"])
                    history["confusion_losses"].append(
                        acc["confusion"] if conf_evaluated else None
                    )
                    mean_bp = sum(bypass_scores) / len(bypass_scores) if bypass_scores else 0.0
                    history["refusal_logodds"].append(mean_bp)

                    if verbose:
                        conf_text = (
                            f"{acc['confusion']:.4f}" if conf_evaluated else "skipped"
                        )
                        print(
                            f"  step {step}: loss={train_loss:.4f} "
                            f"(refusal={acc['refusal']:.4f} score={acc['refusal_score']:.4f} "
                            f"retain={acc['retain']:.4f} confusion={conf_text}) "
                            f"refusal_logodds={mean_bp:.2f}"
                        )

                    # A NaN compares False against everything, so without this guard
                    # one non-finite step would leave lowest_loss NaN and no later
                    # comparison could ever be True again: patience would reset for
                    # ever and neither early stopping nor the LR schedule would fire.
                    if not math.isfinite(train_loss):
                        print(
                            f"  warning: non-finite loss at step {step}; counting it "
                            f"against patience rather than accepting it as an improvement"
                        )
                        patience_counter += 1
                    elif train_loss >= lowest_loss:
                        patience_counter += 1
                    else:
                        lowest_loss = train_loss
                        patience_counter = 0

                    if patience_counter >= patience:
                        if lr_reduce_counter >= n_lr_reduce:
                            if verbose:
                                print(f"  early stop at step {step}")
                            stopped = True
                            break
                        lr_reduce_counter += 1
                        optimizer.param_groups[0]["lr"] /= 10
                        if verbose:
                            print(f"  reducing lr to {optimizer.param_groups[0]['lr']}")
                        patience_counter = 0

                    acc = {k: 0.0 for k in acc}
                    bypass_scores = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            history["n_confusion_steps_skipped"] = n_conf_skipped
            if n_conf_skipped:
                print(
                    f"  warning: the confusion term was skipped on {n_conf_skipped} "
                    f"step(s) because the attack directions were degenerate. Those "
                    f"steps trained without it."
                )
        finally:
            # The tokenizer belongs to the caller, so restore it however the
            # loop exits -- an exception mid-training used to leave it
            # left-padded with a pad token it did not have.
            tokenizer.padding_side = orig_padding_side
            if orig_pad_token_id is None:
                tokenizer.pad_token_id = None
        history["total_time_s"] = time.time() - t0

        if verbose:
            print(f"\ntraining finished in {history['total_time_s']:.1f}s")
            for l in self.target_layers[:5]:
                betas = injector.betas[l].detach().cpu().tolist()
                scales = injector.scales[l].detach().cpu().tolist()
                print(f"  layer {l}: beta={betas}, scale={scales} (fixed)")

        return history

    # ---------------------------------------------------------------- compile

    def compile_to_weights(
        self, hf_model, *, compile_mode: Optional[str] = None
    ) -> Dict[str, object]:
        """Write the fitted decoys into the weights, permanently.

        Edits go to the Hugging Face model, which is the one that gets saved; the
        nnsight wrapper only borrows it for training.

        ``compile_mode`` defaults to the mode training simulated.  Passing a
        different one raises: the two modes leave different weights behind, so the
        decoy would be deployed into a model it was not optimised for.  Choose the
        mode when constructing :class:`DDOOptimizer`.
        """
        if compile_mode is None:
            compile_mode = self.compile_mode
        elif compile_mode != self.compile_mode:
            raise ValueError(
                f"Training simulated {self.compile_mode!r} but compilation was "
                f"asked for {compile_mode!r}. The two modes leave different "
                f"weights behind, so the decoy would be deployed into a model it "
                f"was not optimised for. Construct DDOOptimizer with "
                f"compile_mode={compile_mode!r} instead."
            )

        layers = hf_model.model.layers
        all_indices: Dict[int, List[int]] = {}
        all_info: Dict[int, object] = {}

        for l in self.target_layers:
            lcd = self.layer_compile_data[l]
            neuron_indices = list(lcd["neuron_indices"])
            r_hat = lcd["r_hat"]

            # The readers the injector trained against, not a fresh build: a
            # second build would have to reproduce every argument exactly.
            triggers = self.injector.triggers.get(l)

            layer_indices: List[int] = []
            layer_info: List[object] = []

            # Each decoy is written individually so it carries its own
            # direction, beta and scale.
            for k in range(self.injector.K):
                beta_k = float(self.injector.betas[l][k].detach().item())
                scale_k = float(self.injector.scales[l][k].detach().item())
                dir_k = self.injector.fn_vectors[l][k].detach().cpu().unsqueeze(1)

                r_k = r_hat
                if triggers is not None:
                    r_k = triggers[:, k % triggers.shape[1]]

                idx, info = apply_ddo_to_layer(
                    layers[l],
                    r=r_k,
                    n_decoys=1,
                    beta=beta_k,
                    neuron_indices=[neuron_indices[k]],
                    decoy_scale=scale_k,
                    decoy_dirs_override=dir_k,
                    # The decoy was optimised orthogonal to refusal, not to this
                    # group's reader, so that is what it stays orthogonal to.
                    decoy_orth_to=r_hat,
                    seed=self.seed + l + k * 1000,
                    n_triggers=1,
                    trigger_source="single",
                    neuron_selection=self.neuron_selection,
                    compile_mode=compile_mode,
                )
                layer_indices.extend(idx)
                layer_info.append(info)

            all_indices[l] = layer_indices
            all_info[l] = (
                layer_info[0] if self.injector.K == 1
                else {"multi_rank": True, "K": self.injector.K, "per_neuron": layer_info}
            )

        def _to_py(t: Tensor):
            return t.detach().cpu().tolist()

        return {
            "defense_type": "ddo",
            "compile_mode": compile_mode,
            "target_layers": self.target_layers,
            "skipped_layers": self.skipped_layers,
            "n_decoys": self.n_decoys,
            "K": self.injector.K,
            "n_triggers": self.n_triggers,
            "trigger_source": self.trigger_source,
            "reader_gamma": self.reader_gamma,
            "seed": self.seed,
            "betas": {str(l): _to_py(self.injector.betas[l]) for l in self.target_layers},
            "scales": {str(l): _to_py(self.injector.scales[l]) for l in self.target_layers},
            "neuron_indices": {str(k): v for k, v in all_indices.items()},
            "layer_info": {str(k): v for k, v in all_info.items()},
        }
