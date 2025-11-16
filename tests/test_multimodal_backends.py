import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from nn.llm_multi import MultimodalLlamaModelMultiTokens
from nn.llm_hf_multi import HuggingFaceMultimodalModel


class DummyTokenizer:
    pad_id = 0

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(str(t) for t in token_ids)


class DummyHFModel(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.encoder = nn.Linear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.config = SimpleNamespace(pad_token_id=0, eos_token_id=1)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, **kwargs):
        hidden = torch.tanh(self.encoder(inputs_embeds))
        logits = self.lm_head(hidden)
        return (hidden, logits)

    def generate(self, inputs_embeds=None, max_new_tokens: int = 0, **kwargs):
        batch, seq_len, _ = inputs_embeds.shape
        base = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch, -1)
        new_tokens = torch.zeros(batch, max_new_tokens, dtype=torch.long, device=inputs_embeds.device)
        return torch.cat([base, new_tokens], dim=1)


class DummyLlamaAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)


class DummyLlamaLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = DummyLlamaAttention(dim, n_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Linear(dim, dim)


class DummyLlamaModel(nn.Module):
    def __init__(self, vocab_size: int = 64, dim: int = 16, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.params = SimpleNamespace(dim=dim, n_heads=n_heads, rope_theta=10000, vocab_size=vocab_size)
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([DummyLlamaLayer(dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
        head_dim = dim // n_heads
        self.freqs_cis = torch.ones(128, head_dim // 2, dtype=torch.cfloat)


def _make_dummy_batch(batch_size: int = 2, seq_len: int = 8, vocab_size: int = 32, latent_dim: int = 8):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    spectra = torch.randn(batch_size, latent_dim)
    return {
        'input_ids': input_ids,
        'masked_spectra': spectra,
        'feature_start_indices': torch.zeros(batch_size, dtype=torch.long),
        'answer_start_indices': torch.full((batch_size,), seq_len, dtype=torch.long),
        'mode_mask_single': torch.ones(batch_size, dtype=torch.bool),
        'input_texts': ["prompt"] * batch_size,
        'target_texts': ["target"] * batch_size,
    }


def test_hf_adapter_forward_and_generate():
    torch.manual_seed(0)
    base_model = DummyHFModel()
    wrapper = HuggingFaceMultimodalModel(
        base_model=base_model,
        fm_model=None,
        latent_dim=8,
        hidden_dim=16,
        num_spectral_features=2,
        predict_stellar_params=False,
        enable_classification=False,
        predict_features=False,
        embedding_dim=16,
        vocab_size=32,
        use_cfm=False,
    )
    batch = _make_dummy_batch()
    outputs = wrapper(batch)
    assert 'logits' in outputs
    assert outputs['logits'].shape[0] == batch['input_ids'].shape[0]
    text, _, _, _ = wrapper.generate_response_from_batch(
        batch,
        tokenizer=DummyTokenizer(),
        max_new_tokens=4,
    )
    assert isinstance(text, str)


def test_llama_adapter_forward_and_generate():
    torch.manual_seed(0)
    base_model = DummyLlamaModel()
    wrapper = MultimodalLlamaModelMultiTokens(
        base_model=base_model,
        fm_model=None,
        latent_dim=8,
        hidden_dim=16,
        num_spectral_features=2,
        predict_stellar_params=False,
        enable_classification=False,
        predict_features=False,
        use_checkpoint=False,
        use_cfm=False,
    )
    batch = _make_dummy_batch(vocab_size=base_model.params.vocab_size)
    outputs = wrapper(batch)
    assert 'logits' in outputs
    assert outputs['logits'].shape[0] == batch['input_ids'].shape[0]
    text, _, _, _ = wrapper.generate_response_from_batch(
        batch,
        tokenizer=DummyTokenizer(),
        max_new_tokens=4,
    )
    assert isinstance(text, str)
