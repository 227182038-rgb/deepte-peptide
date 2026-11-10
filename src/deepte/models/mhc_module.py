"""MHC allele encoder modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class MHCAlleleEncoder(nn.Module):
    """Char-level CNN encoder for HLA allele strings.

    Produces a fixed-size embedding that captures the allele's pseudo-sequence
    similarity to known binders.
    """

    def __init__(self, n_chars: int = 49, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(n_chars, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, allele_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        allele_tokens
            (B, L) long tensor of char indices.

        Returns
        -------
        torch.Tensor
            (B, out_dim) allele embedding.
        """
        x = self.embedding(allele_tokens)  # (B, L, embed_dim)
        x = x.transpose(1, 2)  # (B, embed_dim, L)
        x = self.conv(x).squeeze(-1)  # (B, out_dim)
        return self.proj(x)


class MHCSequenceEncoder(nn.Module):
    """Encodes the pseudo-sequence of an HLA allele (34-residue binding groove)."""

    def __init__(
        self, vocab_size: int = 26, d_model: int = 128, n_heads: int = 4, n_layers: int = 2
    ):
        super().__init__()
        from deepte.models.transformer_encoder import (
            SinusoidalPositionalEncoding,
        )

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=64)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, pseudo_seq_tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(pseudo_seq_tokens)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        return x.mean(dim=1)
