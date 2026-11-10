"""DeepTEPeptideNet — the full multi-modal model."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepte.models.transformer_encoder import SequenceTransformer


# ---------------------------------------------------------------------------
# Modality encoders
# ---------------------------------------------------------------------------
class ExpressionEncoder(nn.Module):
    """CNN-LSTM encoder for TE expression signal."""

    def __init__(
        self,
        input_dim: int = 1,
        cnn_channels=(32, 64, 128),
        cnn_kernel: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        n_cancer_types: int = 33,
        cancer_embedding_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cancer_embedding = nn.Embedding(
            n_cancer_types + 1, cancer_embedding_dim, padding_idx=0
        )

        layers = []
        in_ch = input_dim
        for ch in cnn_channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, ch, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = ch
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1] + cancer_embedding_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(lstm_hidden * 2, lstm_hidden)

    def forward(self, expression: torch.Tensor, cancer_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        expression
            Either (B, T, input_dim) for a time-series expression signal,
            or (B, F) for a per-TEP feature vector (in which case F is treated as
            input_dim=1 with T=F).
        cancer_idx
            (B,) cancer-type integer indices.
        """
        cancer_emb = self.cancer_embedding(cancer_idx + 1)  # (B, D)
        if expression.dim() == 2:
            # (B, F) -> (B, F, 1) interpreted as a length-F time series with 1 channel
            x_in = expression.unsqueeze(-1).transpose(1, 2)  # (B, 1, F)
        else:
            x_in = expression.transpose(1, 2)  # (B, in_ch, T)
        x = self.cnn(x_in)  # (B, cnn_channels[-1], T)
        x = x.transpose(1, 2)  # (B, T, cnn_channels[-1])

        # Tile cancer embedding across time
        cancer_emb_expanded = cancer_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, cancer_emb_expanded], dim=-1)

        out, _ = self.lstm(x)  # (B, T, 2*lstm_hidden)
        out = out[:, -1, :]  # take last step (B, 2*H)
        return self.proj(out)


class TranslationEncoder(nn.Module):
    """Set Transformer encoder for per-position Ribo-seq evidence."""

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_inducing_points: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, d_model)

        # Simplified Set Transformer: ISAB-like with inducing points
        self.inducing_points = nn.Parameter(torch.randn(n_inducing_points, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(
        self, ribo_features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        ribo_features
            (B, N, input_dim) per-position Ribo-seq features (variable N).
        mask
            (B, N) bool mask; True = padding.
        """
        b = ribo_features.size(0)
        x = self.proj_in(ribo_features)  # (B, N, d_model)

        # Prepend inducing points
        inducing = self.inducing_points.unsqueeze(0).expand(b, -1, -1)
        x_full = torch.cat([inducing, x], dim=1)  # (B, N+K, d_model)

        # Mask for inducing points is always "not padding"
        if mask is not None:
            cls_mask = torch.zeros(b, inducing.size(1), dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            full_mask = None

        x_full = self.encoder(x_full, src_key_padding_mask=full_mask)
        # Pool over inducing points
        out = x_full[:, : inducing.size(1), :].mean(dim=1)
        return self.proj_out(out)


class TEFeatureEncoder(nn.Module):
    """MLP encoder for TE biophysical / evolutionary features."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims=(128, 64),
        dropout: float = 0.3,
        activation: str = "gelu",
        batch_norm: bool = True,
    ):
        super().__init__()
        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_d = h
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Cross-modal fusion
# ---------------------------------------------------------------------------
class GatedCrossAttentionFusion(nn.Module):
    """Gated cross-attention fusion across four modalities.

    Each pair of modalities exchanges information via cross-attention,
    with learned gates that control the contribution of each fusion step.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.2,
        modality_dropout: float = 0.3,
    ):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.d_model = d_model

        # Project all modalities to d_model
        self.proj_sequence = nn.Linear(d_model, d_model)
        self.proj_expression = nn.Linear(d_model, d_model)
        self.proj_translation = nn.Linear(d_model, d_model)
        self.proj_te_features = nn.Linear(d_model, d_model)

        # Cross-attention blocks (sequence attends to all others)
        self.cross_attns = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                for name in ["expr", "trans", "te_feat"]
            }
        )
        self.gates = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(0.5)) for name in ["expr", "trans", "te_feat"]}
        )
        self.norms = nn.ModuleDict(
            {name: nn.LayerNorm(d_model) for name in ["expr", "trans", "te_feat"]}
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, seq_emb, expr_emb, trans_emb, te_feat_emb):
        """Forward pass.

        All inputs are (B, d_model). Returns (B, d_model) fused representation.
        """
        # Modality dropout: zero out entire modality during training
        if self.training and self.modality_dropout > 0:
            for emb in [expr_emb, trans_emb, te_feat_emb]:
                mask = (
                    torch.rand(emb.size(0), 1, device=emb.device) > self.modality_dropout
                ).float()
                emb = emb * mask

        x = self.proj_sequence(seq_emb)
        kv_dict = {
            "expr": self.proj_expression(expr_emb),
            "trans": self.proj_translation(trans_emb),
            "te_feat": self.proj_te_features(te_feat_emb),
        }

        # Reshape to (B, 1, d_model) for attention
        q = x.unsqueeze(1)
        for name in ["expr", "trans", "te_feat"]:
            kv = kv_dict[name].unsqueeze(1)
            attn_out, _ = self.cross_attns[name](q, kv, kv)
            gate = torch.sigmoid(self.gates[name])
            x = self.norms[name](x + gate * attn_out.squeeze(1))

        x = x + self.ffn(x)
        return self.norm_final(x)


# ---------------------------------------------------------------------------
# Multi-task heads
# ---------------------------------------------------------------------------
class TaskHead(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dims=(64, 32), output_dim: int = 1, dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------
class DeepTEPeptideNet(nn.Module):
    """Full multi-modal DeepTE-Peptide model."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        mcfg = config["model"]
        scfg = mcfg["sequence_encoder"]
        ecfg = mcfg["expression_encoder"]
        tcfg = mcfg["translation_encoder"]
        fcfg = mcfg["te_feature_encoder"]
        fucfg = mcfg["fusion"]
        hcfg = mcfg["heads"]

        # Modality encoders
        self.sequence_encoder = SequenceTransformer(
            vocab_size=scfg["vocab_size"],
            d_model=scfg["d_model"],
            n_heads=scfg["n_heads"],
            n_layers=scfg["n_layers"],
            dropout=scfg["dropout"],
            max_peptide_len=scfg["max_peptide_len"],
            max_mhc_allele_len=scfg["max_mhc_allele_len"],
        )
        self.expression_encoder = ExpressionEncoder(
            input_dim=ecfg["input_dim"],
            cnn_channels=ecfg["cnn_channels"],
            cnn_kernel=ecfg["cnn_kernel"],
            lstm_hidden=ecfg["lstm_hidden"],
            lstm_layers=ecfg["lstm_layers"],
            n_cancer_types=ecfg["n_cancer_types"],
            cancer_embedding_dim=ecfg["cancer_embedding_dim"],
            dropout=ecfg["lstm_dropout"],
        )
        self.translation_encoder = TranslationEncoder(
            input_dim=tcfg["input_dim"],
            d_model=tcfg["d_model"],
            n_heads=tcfg["n_heads"],
            n_layers=tcfg["n_layers"],
            n_inducing_points=tcfg["n_inducing_points"],
            dropout=tcfg["dropout"],
        )
        self.te_feature_encoder = TEFeatureEncoder(
            input_dim=fcfg["input_dim"],
            hidden_dims=fcfg["hidden_dims"],
            dropout=fcfg["dropout"],
        )

        # Project all to fusion d_model
        self.proj_seq = nn.Linear(scfg["d_model"], fucfg["d_model"])
        self.proj_expr = nn.Linear(ecfg["lstm_hidden"], fucfg["d_model"])
        self.proj_trans = nn.Linear(tcfg["d_model"], fucfg["d_model"])
        self.proj_te_feat = nn.Linear(fcfg["hidden_dims"][-1], fucfg["d_model"])

        # Cross-modal fusion
        self.fusion = GatedCrossAttentionFusion(
            d_model=fucfg["d_model"],
            n_heads=fucfg["n_heads"],
            n_layers=fucfg["n_layers"],
            dropout=fucfg["dropout"],
            modality_dropout=fucfg["modality_dropout"],
        )

        # Multi-task heads
        self.head_mhc = TaskHead(fucfg["d_model"], hidden_dims=hcfg["mhc_binding"]["hidden_dims"])
        self.head_tcell = TaskHead(
            fucfg["d_model"], hidden_dims=hcfg["tcell_recognition"]["hidden_dims"]
        )
        self.head_immuno = TaskHead(
            fucfg["d_model"], hidden_dims=hcfg["immunogenicity_score"]["hidden_dims"]
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Sequence encoder
        seq_emb = self.sequence_encoder(
            batch["peptide_tokens"],
            batch["mhc_tokens"],
            peptide_mask=batch.get("peptide_mask"),
            mhc_mask=batch.get("mhc_mask"),
        )

        # Expression encoder — accepts (B, F) per-TEP feature vector
        expr_emb = self.expression_encoder(batch["te_expression"], batch["cancer_idx"])

        # Translation encoder (B, N, 8)
        trans_emb = self.translation_encoder(batch["ribo_features"])

        # TE feature encoder (B, 64)
        te_feat_emb = self.te_feature_encoder(batch["te_features"])

        # Project
        seq_emb = self.proj_seq(seq_emb)
        expr_emb = self.proj_expr(expr_emb)
        trans_emb = self.proj_trans(trans_emb)
        te_feat_emb = self.proj_te_feat(te_feat_emb)

        # Fusion
        fused = self.fusion(seq_emb, expr_emb, trans_emb, te_feat_emb)

        # Heads
        return {
            "mhc_binding": torch.sigmoid(self.head_mhc(fused)).squeeze(-1),
            "tcell_recognition": torch.sigmoid(self.head_tcell(fused)).squeeze(-1),
            "immunogenicity_score": torch.sigmoid(self.head_immuno(fused)).squeeze(-1),
        }
