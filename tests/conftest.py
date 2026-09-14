"""Shared fixtures: tiny random models so the suite runs on CPU in seconds.

A stand-in tokenizer is included so the adapter, the coherence gate, the
fingerprint and the attacks can all be exercised without a network fetch or a
real checkpoint. It is deliberately simple but honest about the interface
:class:`ddo_defense.models.ModelAdapter` actually relies on.
"""

from __future__ import annotations

import pytest
import torch

VOCAB = 128


def _tiny_llama(hidden_act: str = "silu", n_layers: int = 2, d_model: int = 32,
                intermediate: int = 64, vocab: int = VOCAB):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=d_model,
        intermediate_size=intermediate,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act=hidden_act,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).eval()


class FakeTokenizer:
    """Minimal tokenizer covering exactly what ModelAdapter uses.

    Encoding is a deterministic character hash into the vocabulary, so ids are
    always valid for the tiny models above. The chat template has a real suffix
    after the instruction, which is what end-of-instruction token derivation
    depends on.
    """

    SUFFIX = "<|assistant|>"

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self._pad_token = None
        self._pad_token_id = None
        self.padding_side = "right"

    # ``pad_token`` is read and assigned by ModelAdapter, so it must be a real
    # property rather than a plain attribute.
    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    # Assignable, because the library sets it: DDOOptimizer.fit and
    # generate_completions_nnsight both write pad_token_id when it is unset, and
    # fit restores it afterwards.
    @property
    def pad_token_id(self):
        if self._pad_token_id is not None:
            return self._pad_token_id
        return self.eos_token_id if self._pad_token is not None else None

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token_id = value

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            return_tensors=None, **kwargs):
        body = "".join(m["content"] for m in messages)
        text = f"<|user|>{body}{self.SUFFIX}" if add_generation_prompt else f"<|user|>{body}"
        if return_tensors == "pt":
            return torch.tensor([self.encode(text)])
        return text

    def encode(self, text, add_special_tokens=False):
        ids = [(ord(c) % (self.vocab - 3)) + 3 for c in text]
        return ids or [3]

    def decode(self, ids, skip_special_tokens=True):
        try:
            values = ids.tolist()
        except AttributeError:
            values = list(ids)
        return "".join(chr(65 + (int(i) % 26)) for i in values)

    def __call__(self, texts, padding=True, truncation=False, return_tensors="pt",
                 add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self.encode(t)[:64] for t in texts]
        width = max(len(s) for s in seqs)
        input_ids, attention = [], []
        for s in seqs:
            pad = width - len(s)
            # Left padding, matching what ModelAdapter sets and what the
            # last-token conventions throughout the library assume.
            input_ids.append([self.eos_token_id] * pad + s)
            attention.append([0] * pad + [1] * len(s))
        return _Encoding(torch.tensor(input_ids), torch.tensor(attention))


class _Encoding:
    """Mimics the parts of transformers' BatchEncoding that get used."""

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, key):
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}[key]

    def to(self, *_args, **_kwargs):
        return self

    def keys(self):
        return ("input_ids", "attention_mask")


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def tiny_model():
    """Two-layer SwiGLU model with separate gate and up projections."""
    return _tiny_llama()


@pytest.fixture
def tiny_gelu_model():
    """Same shape but a GeGLU activation, as Gemma-2 uses."""
    return _tiny_llama(hidden_act="gelu_pytorch_tanh")


@pytest.fixture
def tiny_fused_model(tiny_model):
    """A model whose MLPs fuse gate and up into one projection, as GLM-4 does."""
    import torch.nn as nn

    for layer in tiny_model.model.layers:
        mlp = layer.mlp
        d_out, d_in = mlp.gate_proj.weight.shape
        fused = nn.Linear(d_in, d_out * 2, bias=False)
        with torch.no_grad():
            fused.weight[:d_out] = mlp.gate_proj.weight
            fused.weight[d_out:] = mlp.up_proj.weight
        mlp.gate_up_proj = fused
        del mlp.gate_proj
        del mlp.up_proj
    return tiny_model


@pytest.fixture
def tiny_adapter(tiny_model, fake_tokenizer):
    from ddo_defense.models import ModelAdapter

    return ModelAdapter(tiny_model, fake_tokenizer)


@pytest.fixture
def tiny_gelu_adapter(tiny_gelu_model):
    from ddo_defense.models import ModelAdapter

    return ModelAdapter(tiny_gelu_model, FakeTokenizer())


@pytest.fixture
def unit_direction(tiny_model):
    """A unit direction the width of the tiny model, not a hardcoded 32."""
    torch.manual_seed(1)
    v = torch.randn(tiny_model.config.hidden_size)
    return v / v.norm()


@pytest.fixture
def unit_directions_per_layer(tiny_model):
    torch.manual_seed(2)
    out = []
    for _ in range(tiny_model.config.num_hidden_layers):
        v = torch.randn(tiny_model.config.hidden_size)
        out.append(v / v.norm())
    return out

def _tiny_gemma2(n_layers: int = 2, d_model: int = 64, intermediate: int = 128,
                 vocab: int = VOCAB):
    """A Gemma-2 shaped model: GeGLU, and a separate feedforward layernorm."""
    pytest.importorskip("transformers")
    from transformers import Gemma2Config, Gemma2ForCausalLM

    config = Gemma2Config(
        vocab_size=vocab,
        hidden_size=d_model,
        intermediate_size=intermediate,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return Gemma2ForCausalLM(config).eval()


@pytest.fixture
def tiny_gemma2_model():
    return _tiny_gemma2()
