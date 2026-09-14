"""One tokenizer-driven adapter for every supported model family.

Prompt formatting is derived from the tokenizer that ships with the checkpoint,
so no per-family chat template is maintained here:

* **Prompt formatting** comes from ``tokenizer.apply_chat_template`` with
  ``add_generation_prompt=True``, so it always matches what the model was tuned
  on.  Reasoning models are asked not to emit a thinking block.
* **Refusal tokens** come from a small pinned per-family table when the
  checkpoint path matches a known family, and are derived from the checkpoint's
  own vocabulary for anything unrecognised.

Use one adapter for training, direction estimation and attacks, so all three
share a single formatting path and a decoy is fitted against exactly the prompts
an attacker will produce.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


#: Words whose first token indicates the model is beginning a refusal.
DEFAULT_REFUSAL_WORDS: Sequence[str] = ("I",)

#: Prefixes used by the first-token anchoring term, which is a margin between
#: the strongest refusal logit and the strongest compliance logit. These are a
#: separate concern from ``refusal_toks`` above, which identifies the single
#: refusal token used for direction estimation and monitoring.
ANCHOR_REFUSAL_WORDS: Sequence[str] = ("I", "Sorry", "cannot")
ANCHOR_COMPLIANCE_WORDS: Sequence[str] = ("Sure", "Here")

#: Refusal token ids per model family, keyed by a substring of the checkpoint
#: path.  :meth:`ModelAdapter.from_pretrained` uses the entry for a recognised
#: family, and derives the ids from the tokenizer for anything unrecognised.
FAMILY_REFUSAL_TOKS: Dict[str, List[int]] = {
    "llama-3": [40],
    "llama-2": [306],
    "gemma": [235285],
    "qwen": [40],
    "yi": [59597],
    "mistral": [315],
    "glm": [74],
}


def family_refusal_toks(model_path: str) -> Optional[List[int]]:
    """Refusal ids for a recognised family, or ``None`` if the path is unknown."""
    lowered = (model_path or "").lower()
    for key, ids in FAMILY_REFUSAL_TOKS.items():
        if key in lowered:
            return list(ids)
    # "llama" without a version suffix is ambiguous; leave it to the tokenizer.
    return None


class ModelAdapter:
    """Uniform access to a causal LM's prompt format, tokens and submodules.

    Construct either from a checkpoint via :meth:`from_pretrained`, or around an
    already-loaded model by passing ``model`` and ``tokenizer`` directly.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        refusal_words: Sequence[str] = DEFAULT_REFUSAL_WORDS,
        refusal_toks: Optional[Sequence[int]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._refusal_words = tuple(refusal_words)
        self._refusal_toks_override = list(refusal_toks) if refusal_toks else None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.refusal_toks = (
            self._refusal_toks_override
            if self._refusal_toks_override is not None
            else self._derive_refusal_toks()
        )

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | Dict = "auto",
        trust_remote_code: bool = False,
        refusal_words: Sequence[str] = DEFAULT_REFUSAL_WORDS,
        refusal_toks: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> "ModelAdapter":
        """Load a checkpoint and wrap it.

        ``trust_remote_code`` defaults to ``False`` because enabling it executes
        code from the checkpoint.  A few checkpoints need it; prefer a
        transformers-native conversion where one exists (for GLM-4, use
        ``THUDM/glm-4-9b-chat-hf``, which loads as ``GlmForCausalLM``).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if refusal_toks is None and tuple(refusal_words) == tuple(DEFAULT_REFUSAL_WORDS):
            # The pinned family ids are a shortcut for the default words only.
            # An explicit refusal_words must win, or it would be accepted and
            # then silently ignored.
            refusal_toks = family_refusal_toks(model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()
        model.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        return cls(
            model, tokenizer, refusal_words=refusal_words, refusal_toks=refusal_toks
        )

    # ------------------------------------------------------- prompt formatting

    def format_instruction(self, instruction: str, output: Optional[str] = None) -> str:
        """Render one instruction into the model's own chat format."""
        messages = [{"role": "user", "content": instruction}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Tokenizers whose template does not accept enable_thinking.
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            # Falling back to the bare instruction here would format every prompt
            # with no chat wrapper, and silently: direction estimation, the decoy
            # and the attack would all be measured on text the model was never
            # trained to answer, and nothing downstream could tell.
            raise RuntimeError(
                f"This checkpoint's tokenizer cannot render a chat template "
                f"({type(exc).__name__}: {exc}). Every prompt in this library goes "
                f"through it, so there is no sensible fallback -- an unwrapped "
                f"prompt would silently change what is being measured. Supply a "
                f"tokenizer with a chat_template, or pass format_fn explicitly."
            ) from exc

        if output is not None:
            text = text + output
        return text

    def tokenize(
        self,
        instructions: Sequence[str],
        outputs: Optional[Sequence[str]] = None,
    ):
        """Batch-tokenize instructions, left-padded, ready for the model."""
        if outputs is not None:
            prompts = [
                self.format_instruction(i, o) for i, o in zip(instructions, outputs)
            ]
        else:
            prompts = [self.format_instruction(i) for i in instructions]

        # ``apply_chat_template`` renders whatever special tokens the template
        # asks for, and the Llama-2, Llama-3, Mistral and Gemma templates all
        # emit ``bos_token`` themselves.  Tokenizing that text with the default
        # ``add_special_tokens=True`` would prepend a second BOS, shifting every
        # position by one on those families.  Decide from the rendered text, so
        # a tokenizer whose template emits no BOS still gets one.
        bos = getattr(self.tokenizer, "bos_token", None)
        already_has_bos = bool(bos) and all(p.startswith(bos) for p in prompts)

        return self.tokenizer(
            prompts, padding=True, truncation=False, return_tensors="pt",
            add_special_tokens=not already_has_bos,
        )

    # The direction and attack helpers take a tokenizing callable, not the
    # adapter, so this exposes ``tokenize`` in the shape they expect.
    @property
    def tokenize_instructions_fn(self):
        def _fn(instructions, outputs=None):
            return self.tokenize(instructions, outputs)

        return _fn

    def anchor_toks(self) -> tuple:
        """``(refusal_ids, compliance_ids)`` for the first-token margin term."""
        return (
            self._first_token_ids(ANCHOR_REFUSAL_WORDS),
            self._first_token_ids(ANCHOR_COMPLIANCE_WORDS),
        )

    def _first_token_ids(self, words: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for word in words:
            enc = self.tokenizer.encode(word, add_special_tokens=False)
            if enc and int(enc[0]) not in ids:
                ids.append(int(enc[0]))
        return ids

    def _derive_refusal_toks(self) -> List[int]:
        """First-token ids of the configured refusal words."""
        ids: List[int] = []
        for word in self._refusal_words:
            enc = self.tokenizer.encode(word, add_special_tokens=False)
            if enc:
                ids.append(int(enc[0]))
        # Preserve order while removing duplicates.
        seen, out = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    # ------------------------------------------------------------- structure

    @property
    def config(self):
        return self.model.config

    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def blocks(self):
        """The decoder layer list, whatever the architecture calls it."""
        for path in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
            obj = self.model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise ValueError(
            "Could not locate the decoder layers on this model. Expected "
            "model.model.layers, model.transformer.h or model.gpt_neox.layers"
        )

    @property
    def attn_modules(self):
        mods = []
        for b in self.blocks:
            attn = next(
                (getattr(b, name) for name in ("self_attn", "attn", "attention")
                 if getattr(b, name, None) is not None),
                None,
            )
            if attn is None:
                raise ValueError(
                    "Could not locate the attention submodule on this block. "
                    "Expected self_attn, attn or attention."
                )
            mods.append(attn)
        return torch.nn.ModuleList(mods)

    @property
    def mlp_modules(self):
        return torch.nn.ModuleList([b.mlp for b in self.blocks])

    def activation_fn(self, layer_idx: int = 0):
        """The activation actually used by a layer's MLP."""
        from ddo_defense.mlp import get_activation_fn

        return get_activation_fn(self.blocks[layer_idx], self.config)

    def gate_arch(self) -> str:
        """``"swiglu"``, ``"geglu"`` or ``"fused_gate_up"``."""
        from ddo_defense.mlp import is_fused_glu

        if is_fused_glu(self.blocks[0]):
            return "fused_gate_up"
        act = str(getattr(self.config, "hidden_act", "") or
                  getattr(self.config, "hidden_activation", "")).lower()
        return "geglu" if "gelu" in act else "swiglu"

    # ------------------------------------------------------------ generation

    def generate(
        self,
        instructions: Sequence[str],
        *,
        max_new_tokens: int = 64,
        batch_size: int = 8,
        fwd_pre_hooks: Sequence = (),
        fwd_hooks: Sequence = (),
    ) -> List[str]:
        """Greedy-decode a list of instructions, optionally under hooks."""
        from ddo_defense.hooks import add_hooks

        outs: List[str] = []
        for i in range(0, len(instructions), batch_size):
            batch = list(instructions[i:i + batch_size])
            enc = self.tokenize(batch)
            with torch.no_grad():
                with add_hooks(
                    module_forward_pre_hooks=list(fwd_pre_hooks),
                    module_forward_hooks=list(fwd_hooks),
                ):
                    gen = self.model.generate(
                        input_ids=enc.input_ids.to(self.model.device),
                        attention_mask=enc.attention_mask.to(self.model.device),
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            gen = gen[:, enc.input_ids.shape[-1]:]
            outs.extend(
                self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen
            )
        return outs

    def generate_completions(
        self,
        dataset: Sequence[Dict],
        *,
        max_new_tokens: int = 64,
        batch_size: int = 8,
        fwd_pre_hooks: Sequence = (),
        fwd_hooks: Sequence = (),
    ) -> List[Dict[str, str]]:
        """Greedy-decode dataset records into ``{category, prompt, response}``."""
        instructions = [x["instruction"] for x in dataset]
        responses = self.generate(
            instructions,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
        )
        return [
            {
                "category": rec.get("category"),
                "prompt": rec["instruction"],
                "response": resp,
            }
            for rec, resp in zip(dataset, responses)
        ]

    def free(self) -> None:
        """Drop the model and release GPU memory."""
        import gc

        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
