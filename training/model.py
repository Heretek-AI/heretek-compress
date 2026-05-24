"""Byte-level autoregressive transformer mirroring the Rust heretik architecture.

Decoder-only transformer with:
- Token embedding (vocab_size=256) + positional embedding (context_window=512)
- 8 decoder layers: causal self-attention + pre-norm residuals + GELU FFN
- Final layer norm + 256-way softmax output

Tensor naming matches candle VarBuilder hierarchy for direct safetensors
interop with Transformer::load_from_buffer in heretik-predictor.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(NamedTuple):
    num_layers: int = 8
    embed_dim: int = 384
    num_heads: int = 6
    context_window: int = 512
    vocab_size: int = 256

    def to_dict(self) -> dict:
        d = self._asdict()
        # Export context_window with +1 to account for BOS token position.
        # The Rust side loads with this value to match the pos_embed table size.
        d["context_window"] = self.context_window + 1
        return d

    @property
    def pos_table_size(self) -> int:
        """Size of the positional embedding table (needs +1 for BOS)."""
        return self.context_window + 1


class DecoderLayer(nn.Module):
    """Single decoder layer: causal self-attention + FFN with pre-norm residuals.

    Uses a combined QKV projection (embed_dim → 3*embed_dim) to match the
    Rust architecture exactly.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined QKV projection: embed_dim → 3 * embed_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        # Output projection after attention mixture
        self.out = nn.Linear(embed_dim, embed_dim, bias=True)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        # FFN: embed_dim → 4*embed_dim → embed_dim
        self.ff_up = nn.Linear(embed_dim, 4 * embed_dim, bias=True)
        self.ff_down = nn.Linear(4 * embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, embed_dim = x.shape

        # ---- Self-attention -------------------------------------------------
        qkv = self.qkv(x)  # (seq_len, 3 * embed_dim)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)  # (3, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (num_heads, seq_len, head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (num_heads, seq_len, seq_len)
        scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (num_heads, seq_len, head_dim)

        # Merge heads: (num_heads, seq_len, head_dim) → (seq_len, embed_dim)
        attn_out = attn_out.permute(1, 0, 2).reshape(seq_len, embed_dim)
        attn_out = self.out(attn_out)

        # Residual + norm
        x = x + attn_out
        x = self.norm1(x)

        # ---- FFN ------------------------------------------------------------
        ff = self.ff_up(x)
        ff = F.gelu(ff)  # erf-based, same as candle-nn gelu()
        ff = self.ff_down(ff)

        # Residual + norm
        x = x + ff
        x = self.norm2(x)

        return x


class Transformer(nn.Module):
    """Byte-level autoregressive transformer (decoder-only).

    Designed for direct safetensors interop with heretik-predictor's
    Transformer::load_from_buffer. Tensor names in the export match the
    candle VarBuilder hierarchy.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        # pos_table_size = context_window + 1 for BOS token
        self.pos_embed = nn.Embedding(config.pos_table_size, config.embed_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(config.embed_dim, config.num_heads)
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.embed_dim, eps=1e-5)
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: LongTensor of shape (seq_len,) — token indices.

        Returns:
            Probability distribution tensor of shape (seq_len, vocab_size).
            Each row sums to 1.0.
        """
        seq_len = tokens.shape[0]
        device = tokens.device

        positions = torch.arange(seq_len, device=device, dtype=torch.long)

        tok_emb = self.token_embed(tokens)  # (seq_len, embed_dim)
        pos_emb = self.pos_embed(positions)  # (seq_len, embed_dim)
        x = tok_emb + pos_emb

        # Causal mask: (1, seq_len, seq_len) with -inf in upper triangle
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)  # upper triangle (excl. diag) = -inf
        mask = mask.unsqueeze(0)  # (1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)
        logits = self.output(x)  # (seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1)

        return probs

    def _tensor_dict(self) -> dict[str, torch.Tensor]:
        """Build a dict of {candle_tensor_name: tensor} for safetensors export.

        Matches the VarBuilder pp() hierarchy so load_from_buffer reconstructs
        the model exactly.
        """
        tensors: dict[str, torch.Tensor] = {}

        # Embeddings
        tensors["token_embed.weight"] = self.token_embed.weight.data
        tensors["pos_embed.weight"] = self.pos_embed.weight.data

        # Decoder layers
        for i, layer in enumerate(self.layers):
            prefix = f"layer_{i}"
            tensors[f"{prefix}.qkv.weight"] = layer.qkv.weight.data
            tensors[f"{prefix}.qkv.bias"] = layer.qkv.bias.data
            tensors[f"{prefix}.out.weight"] = layer.out.weight.data
            tensors[f"{prefix}.out.bias"] = layer.out.bias.data
            tensors[f"{prefix}.norm1.weight"] = layer.norm1.weight.data
            tensors[f"{prefix}.norm1.bias"] = layer.norm1.bias.data
            tensors[f"{prefix}.norm2.weight"] = layer.norm2.weight.data
            tensors[f"{prefix}.norm2.bias"] = layer.norm2.bias.data
            tensors[f"{prefix}.ff_up.weight"] = layer.ff_up.weight.data
            tensors[f"{prefix}.ff_up.bias"] = layer.ff_up.bias.data
            tensors[f"{prefix}.ff_down.weight"] = layer.ff_down.weight.data
            tensors[f"{prefix}.ff_down.bias"] = layer.ff_down.bias.data

        # Final norm + output projection
        tensors["final_norm.weight"] = self.final_norm.weight.data
        tensors["final_norm.bias"] = self.final_norm.bias.data
        tensors["output.weight"] = self.output.weight.data
        tensors["output.bias"] = self.output.bias.data

        return tensors
