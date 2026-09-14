"""Over-refusal repair: both methods, and what each one touches."""

from __future__ import annotations

import pytest
import torch

from ddo_defense.defense.debias import (
    apply_lm_head_debiasing,
    apply_projection_debiasing,
    default_layers,
)


class _FakeTokenizer:
    """Maps each word to a distinct id, enough to exercise the edit."""

    def __init__(self):
        self._vocab = {}

    def encode(self, text, add_special_tokens=False):
        if text not in self._vocab:
            self._vocab[text] = len(self._vocab) + 1
        return [self._vocab[text]]


class _Adapter:
    def __init__(self, model):
        self.model = model
        self.tokenizer = _FakeTokenizer()
        self.n_layers = model.config.num_hidden_layers
        self.d_model = model.config.hidden_size

    @property
    def blocks(self):
        return self.model.model.layers


def test_default_layers_are_late_and_in_range():
    assert default_layers(32) == [24, 26, 28, 30]
    for depth in (8, 16, 42, 80):
        layers = default_layers(depth)
        assert all(0 <= l < depth for l in layers)
        assert min(layers) > depth // 2


def test_lm_head_edit_only_touches_named_rows(tiny_model, unit_direction):
    adapter = _Adapter(tiny_model)
    before = tiny_model.lm_head.weight.data.clone()

    info = apply_lm_head_debiasing(
        adapter, unit_direction, boost=0.1, suppress=0.05,
        compliance_words=("The", "Here"), refusal_words=("I",),
    )

    touched = set(info["boosted_token_ids"]) | set(info["suppressed_token_ids"])
    assert len(touched) == 3

    after = tiny_model.lm_head.weight.data
    changed = {i for i in range(after.shape[0]) if not torch.allclose(after[i], before[i])}
    assert changed == touched


def test_lm_head_boost_and_suppress_move_opposite_ways(tiny_model, unit_direction):
    adapter = _Adapter(tiny_model)
    before = tiny_model.lm_head.weight.data.clone()

    info = apply_lm_head_debiasing(
        adapter, unit_direction, boost=0.2, suppress=0.1,
        compliance_words=("The",), refusal_words=("I",),
    )

    v = unit_direction.to(tiny_model.lm_head.weight.dtype)
    boosted = info["boosted_token_ids"][0]
    suppressed = info["suppressed_token_ids"][0]
    after = tiny_model.lm_head.weight.data

    assert torch.allclose(after[boosted] - before[boosted], 0.2 * v, atol=1e-5)
    assert torch.allclose(after[suppressed] - before[suppressed], -0.1 * v, atol=1e-5)


def test_projection_covers_every_residual_writer(tiny_model, unit_direction):
    """Embedding, attention output and MLP output are all covered by default."""
    adapter = _Adapter(tiny_model)
    info = apply_projection_debiasing(adapter, unit_direction)

    assert info["include_embedding"] is True
    assert info["include_attn_out"] is True
    assert info["n_matrices_edited"] >= 2 * tiny_model.config.num_hidden_layers

    v = unit_direction.float()
    for layer in tiny_model.model.layers:
        assert (v @ layer.mlp.down_proj.weight.data.float()).abs().max() < 1e-3
        assert (v @ layer.self_attn.o_proj.weight.data.float()).abs().max() < 1e-3
    assert (tiny_model.get_input_embeddings().weight.data.float() @ v).abs().max() < 1e-3


def test_individual_matrices_can_be_excluded(tiny_model, unit_direction):
    adapter = _Adapter(tiny_model)
    info = apply_projection_debiasing(
        adapter, unit_direction, include_embedding=False, include_attn_out=False
    )
    assert info["n_matrices_edited"] == tiny_model.config.num_hidden_layers


def test_projection_is_idempotent(tiny_model, unit_direction):
    adapter = _Adapter(tiny_model)
    apply_projection_debiasing(adapter, unit_direction)
    once = tiny_model.model.layers[0].mlp.down_proj.weight.data.clone()
    apply_projection_debiasing(adapter, unit_direction)
    twice = tiny_model.model.layers[0].mlp.down_proj.weight.data
    assert torch.allclose(once, twice, atol=1e-5)


def test_lm_head_missing_is_reported(unit_direction):
    class _Headless:
        config = type("C", (), {"num_hidden_layers": 1, "hidden_size": 32})()

    class _A:
        model = _Headless()
        tokenizer = _FakeTokenizer()

    with pytest.raises(ValueError, match="lm_head"):
        apply_lm_head_debiasing(_A(), unit_direction)


# --- tied embeddings ---------------------------------------------------------
# Gemma-2 ties the input embedding to the output head by default, and Gemma-2-9B-it
# is a supported model. An edit to either matrix is then an edit to both, which
# changes what each method does.

def _tied_model(d_model: int = 32, vocab: int = 64):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=d_model, intermediate_size=2 * d_model,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_tying_is_detected(tiny_model):
    from ddo_defense.defense.debias import weights_are_tied

    assert weights_are_tied(_tied_model()) is True
    assert weights_are_tied(tiny_model) is False


def test_projection_leaves_a_tied_embedding_alone(fake_tokenizer):
    """Projecting a tied embedding would strip the direction from every logit."""
    from ddo_defense.defense.debias import apply_projection_debiasing
    from ddo_defense.models import ModelAdapter

    model = _tied_model()
    adapter = ModelAdapter(model, fake_tokenizer)
    before = model.get_input_embeddings().weight.data.clone()

    v = torch.randn(model.config.hidden_size)
    info = apply_projection_debiasing(adapter, v / v.norm())

    assert info["tied_embeddings"] is True
    assert info["embedding_skipped_because_tied"] is True
    assert "embed_tokens" not in info["edited"]
    assert torch.equal(model.get_input_embeddings().weight.data, before)
    # The write paths were still projected.
    assert any("o_proj" in name for name in info["edited"])
    assert any("down_proj" in name for name in info["edited"])


def test_projection_still_edits_an_untied_embedding(tiny_model, fake_tokenizer):
    from ddo_defense.defense.debias import apply_projection_debiasing
    from ddo_defense.models import ModelAdapter

    adapter = ModelAdapter(tiny_model, fake_tokenizer)
    before = tiny_model.get_input_embeddings().weight.data.clone()

    v = torch.randn(tiny_model.config.hidden_size)
    info = apply_projection_debiasing(adapter, v / v.norm())

    assert info["tied_embeddings"] is False
    assert "embed_tokens" in info["edited"]
    assert not torch.equal(tiny_model.get_input_embeddings().weight.data, before)


def test_lm_head_method_reports_tying(fake_tokenizer):
    from ddo_defense.defense.debias import apply_lm_head_debiasing
    from ddo_defense.models import ModelAdapter

    model = _tied_model()
    adapter = ModelAdapter(model, fake_tokenizer)
    v = torch.randn(model.config.hidden_size)

    info = apply_lm_head_debiasing(adapter, v / v.norm())
    assert info["tied_embeddings"] is True
