"""The single tokenizer-driven adapter every other module formats prompts through.

This is the module a transformers upgrade is most likely to break, and the one
whose consistency the whole design depends on: training, direction estimation and
attacks must all format prompts the same way or the decoy is fitted against
prompts the attack never produces.
"""

from __future__ import annotations

import pytest
import torch

from ddo_defense.models import DEFAULT_REFUSAL_WORDS, ModelAdapter

from tests.conftest import VOCAB, FakeTokenizer


def test_refusal_tokens_are_derived_from_the_tokenizer(tiny_adapter, fake_tokenizer):
    assert tiny_adapter.refusal_toks
    expected = fake_tokenizer.encode(DEFAULT_REFUSAL_WORDS[0])[0]
    assert tiny_adapter.refusal_toks[0] == expected


def test_refusal_words_are_configurable(tiny_model, fake_tokenizer):
    adapter = ModelAdapter(tiny_model, fake_tokenizer, refusal_words=("I", "Sorry"))
    assert len(adapter.refusal_toks) >= 1


def test_padding_is_forced_left(tiny_adapter):
    """Last-token conventions throughout the library require left padding."""
    assert tiny_adapter.tokenizer.padding_side == "left"
    assert tiny_adapter.tokenizer.pad_token is not None


def test_format_instruction_wraps_the_instruction(tiny_adapter):
    text = tiny_adapter.format_instruction("how do I bake bread")
    assert "how do I bake bread" in text
    assert text.endswith(tiny_adapter.tokenizer.SUFFIX)


def test_format_instruction_appends_a_target(tiny_adapter):
    text = tiny_adapter.format_instruction("question", "ANSWER")
    assert text.endswith("ANSWER")


def test_tokenize_pads_a_ragged_batch(tiny_adapter):
    enc = tiny_adapter.tokenize(["short", "a much longer instruction than the first"])
    assert enc.input_ids.shape[0] == 2
    assert enc.input_ids.shape == enc.attention_mask.shape
    # Left padding means the shorter row carries the zeros at the front.
    assert enc.attention_mask[0, 0] == 0
    assert enc.attention_mask[0, -1] == 1


def test_structure_accessors_match_the_config(tiny_adapter, tiny_model):
    assert tiny_adapter.n_layers == tiny_model.config.num_hidden_layers
    assert tiny_adapter.d_model == tiny_model.config.hidden_size
    assert len(tiny_adapter.blocks) == tiny_adapter.n_layers
    assert len(tiny_adapter.attn_modules) == tiny_adapter.n_layers
    assert len(tiny_adapter.mlp_modules) == tiny_adapter.n_layers


def test_gate_arch_detection(tiny_adapter, tiny_gelu_adapter):
    assert tiny_adapter.gate_arch() == "swiglu"
    assert tiny_gelu_adapter.gate_arch() == "geglu"


def test_gate_arch_detects_fused(tiny_fused_model, fake_tokenizer):
    adapter = ModelAdapter(tiny_fused_model, fake_tokenizer)
    assert adapter.gate_arch() == "fused_gate_up"


def test_activation_fn_follows_the_model(tiny_adapter, tiny_gelu_adapter):
    x = torch.linspace(-2, 2, 8)
    silu = tiny_adapter.activation_fn(0)
    gelu = tiny_gelu_adapter.activation_fn(0)
    assert torch.allclose(silu(x), torch.nn.functional.silu(x), atol=1e-5)
    assert not torch.allclose(gelu(x), torch.nn.functional.silu(x), atol=1e-3)


def test_generate_returns_one_string_per_prompt(tiny_adapter):
    out = tiny_adapter.generate(["a", "b", "c"], max_new_tokens=3, batch_size=2)
    assert len(out) == 3
    assert all(isinstance(s, str) for s in out)


def test_generate_completions_preserves_records(tiny_adapter):
    dataset = [
        {"instruction": "first", "category": "x"},
        {"instruction": "second", "category": "y"},
    ]
    out = tiny_adapter.generate_completions(dataset, max_new_tokens=2, batch_size=2)
    assert [r["prompt"] for r in out] == ["first", "second"]
    assert [r["category"] for r in out] == ["x", "y"]
    assert all("response" in r for r in out)


def test_missing_layers_is_reported_clearly(fake_tokenizer):
    class _Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"num_hidden_layers": 1, "hidden_size": 4})()

    adapter = ModelAdapter.__new__(ModelAdapter)
    adapter.model = _Bare()
    with pytest.raises(ValueError, match="decoder layers"):
        _ = adapter.blocks


# --- per-family refusal ids -------------------------------------------------
# Refusal token ids are looked up from the checkpoint path for recognised
# families, and derived from the tokenizer for anything unrecognised.

def test_known_families_resolve_to_their_pinned_ids():
    from ddo_defense.models import family_refusal_toks

    cases = {
        "meta-llama/Meta-Llama-3-8B-Instruct": [40],
        "meta-llama/Llama-2-7b-chat-hf": [306],
        "google/gemma-2-9b-it": [235285],
        "Qwen/Qwen3-8B": [40],
        "01-ai/Yi-1.5-9B-Chat": [59597],
        "mistralai/Mistral-7B-Instruct-v0.3": [315],
        "THUDM/glm-4-9b-chat-hf": [74],
    }
    for path, expected in cases.items():
        assert family_refusal_toks(path) == expected, path


def test_matching_is_case_insensitive():
    from ddo_defense.models import family_refusal_toks

    assert family_refusal_toks("META-LLAMA/META-LLAMA-3-8B-INSTRUCT") == [40]
    assert family_refusal_toks("01-AI/YI-1.5-9B-CHAT") == [59597]


def test_an_unrecognised_path_defers_to_the_tokenizer():
    from ddo_defense.models import family_refusal_toks

    assert family_refusal_toks("some-org/some-new-model") is None
    assert family_refusal_toks("") is None
    assert family_refusal_toks(None) is None


def test_a_local_directory_path_still_matches():
    """Defended checkpoints are saved to paths that keep the family name."""
    from ddo_defense.models import family_refusal_toks

    assert family_refusal_toks("/scratch/runs/llama-3-defended/model") == [40]


def test_an_explicit_override_wins_over_derivation(tiny_model, fake_tokenizer):
    from ddo_defense.models import ModelAdapter

    adapter = ModelAdapter(tiny_model, fake_tokenizer, refusal_toks=[1234])
    assert adapter.refusal_toks == [1234]


def test_without_an_override_the_tokenizer_is_used(tiny_model, fake_tokenizer):
    from ddo_defense.models import ModelAdapter

    adapter = ModelAdapter(tiny_model, fake_tokenizer)
    assert adapter.refusal_toks == [fake_tokenizer.encode("I")[0]]


# --- special tokens ----------------------------------------------------------
# apply_chat_template renders the template's own special tokens. Tokenizing that
# string again with add_special_tokens=True prepends a second BOS on every
# family whose template emits one, which is Llama-2, Llama-3, Mistral and Gemma.

class _BosTokenizer(FakeTokenizer):
    """A stub whose chat template emits BOS, as the real ones do."""

    BOS = "<s>"

    def __init__(self, vocab=VOCAB):
        super().__init__(vocab)
        self.bos_token = self.BOS

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return self.BOS + super().apply_chat_template(messages, tokenize=tokenize, **kwargs)

    def __call__(self, texts, padding=True, truncation=False, return_tensors="pt",
                 add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        if add_special_tokens:
            texts = [self.BOS + t for t in texts]
        self.last_texts = list(texts)
        return super().__call__(texts, padding=padding, truncation=truncation,
                                return_tensors=return_tensors)


def test_a_template_that_emits_bos_does_not_get_a_second_one(tiny_model):
    tok = _BosTokenizer()
    adapter = ModelAdapter(tiny_model, tok)

    adapter.tokenize(["hello"])
    rendered = tok.last_texts[0]

    assert rendered.startswith(_BosTokenizer.BOS)
    assert not rendered.startswith(_BosTokenizer.BOS * 2), (
        f"prompt carries two BOS tokens: {rendered[:40]!r}"
    )


def test_a_template_without_bos_still_gets_one(tiny_model):
    """The decision is read off the rendered text, not hardcoded off."""
    class _NoBosTemplate(_BosTokenizer):
        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            return FakeTokenizer.apply_chat_template(self, messages, tokenize=tokenize, **kwargs)

    tok = _NoBosTemplate()
    adapter = ModelAdapter(tiny_model, tok)

    adapter.tokenize(["hello"])
    assert tok.last_texts[0].startswith(_BosTokenizer.BOS)
