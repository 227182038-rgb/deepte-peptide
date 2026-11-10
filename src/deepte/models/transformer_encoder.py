"""Transformer encoder for peptide sequences."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SequenceTransformer(nn.Module):
    """Transformer encoder for peptide + MHC allele sequences.

    The peptide and MHC allele are tokenized separately, concatenated with a
    <CLS> token, and processed through a shared transformer encoder.
    """

    def __init__(
        self,
        vocab_size: int = 26,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_peptide_len: int = 25,
        max_mhc_allele_len: int = 30,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_peptide_len = max_peptide_len
        self.max_mhc_allele_len = max_mhc_allele_len

        self.peptide_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # MHC allele: char-level CNN encoder
        self.mhc_char_embedding = nn.Embedding(49, d_model, padding_idx=0)

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_len=max_peptide_len + max_mhc_allele_len + 2
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        peptide_tokens: torch.Tensor,
        mhc_allele_tokens: torch.Tensor,
        peptide_mask: torch.Tensor | None = None,
        mhc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        peptide_tokens
            (B, L_pep) int tensor.
        mhc_allele_tokens
            (B, L_mhc) int tensor (char-level).
        peptide_mask
            (B, L_pep) bool tensor; True = padding (ignored).
        mhc_mask
            (B, L_mhc) bool tensor; True = padding.

        Returns
        -------
        torch.Tensor
            (B, d_model) CLS-pooled representation.
        """
        b = peptide_tokens.size(0)

        pep_emb = self.peptide_embedding(peptide_tokens) * math.sqrt(self.d_model)
        mhc_emb = self.mhc_char_embedding(mhc_allele_tokens) * math.sqrt(self.d_model)

        # Expand CLS token to batch
        cls = self.cls_token.expand(b, -1, -1)

        # Concatenate: [CLS] [peptide] [MHC allele]
        x = torch.cat([cls, pep_emb, mhc_emb], dim=1)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Build padding mask (True = padding positions to be ignored by attention)
        cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=x.device)
        if peptide_mask is None:
            peptide_mask = peptide_tokens == 0
        if mhc_mask is None:
            mhc_mask = mhc_allele_tokens == 0
        full_mask = torch.cat([cls_mask, peptide_mask, mhc_mask], dim=1)

        # TransformerEncoder expects src_key_padding_mask where True = ignore
        x = self.transformer(x, src_key_padding_mask=full_mask)

        # Pool via CLS token
        return x[:, 0, :]
