"""Unit tests for model architectures."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepte.models.te_peptide_net import (
    DeepTEPeptideNet,
    ExpressionEncoder,
    GatedCrossAttentionFusion,
    TEFeatureEncoder,
    TranslationEncoder,
)
from deepte.models.transformer_encoder import SequenceTransformer


@pytest.fixture
def config():
    return {
        "model": {
            "sequence_encoder": {
                "vocab_size": 26, "d_model": 64, "n_heads": 4, "n_layers": 2,
                "dropout": 0.1, "max_peptide_len": 25, "max_mhc_allele_len": 30,
            },
            "expression_encoder": {
                "input_dim": 1, "cnn_channels": [16, 32], "cnn_kernel": 3,
                "lstm_hidden": 32, "lstm_layers": 1, "cancer_embedding_dim": 16,
                "lstm_dropout": 0.1, "n_cancer_types": 33,
            },
            "translation_encoder": {
                "input_dim": 8, "d_model": 32, "n_heads": 4, "n_layers": 1,
                "n_inducing_points": 8, "dropout": 0.1,
            },
            "te_feature_encoder": {
                "input_dim": 64, "hidden_dims": [64, 32], "dropout": 0.1,
                "activation": "gelu", "batch_norm": True,
            },
            "fusion": {
                "d_model": 64, "n_heads": 4, "n_layers": 1,
                "dropout": 0.1, "modality_dropout": 0.3,
            },
            "heads": {
                "mhc_binding": {"hidden_dims": [32, 16], "output_dim": 1},
                "tcell_recognition": {"hidden_dims": [32, 16], "output_dim": 1},
                "immunogenicity_score": {"hidden_dims": [32, 16], "output_dim": 1},
            },
        }
    }


class TestSequenceTransformer:
    def test_forward(self, config):
        cfg = config["model"]["sequence_encoder"]
        model = SequenceTransformer(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            max_peptide_len=cfg["max_peptide_len"],
            max_mhc_allele_len=cfg["max_mhc_allele_len"],
        )
        B = 4
        pep = torch.randint(0, 26, (B, 25))
        mhc = torch.randint(0, 49, (B, 30))
        out = model(pep, mhc)
        assert out.shape == (B, cfg["d_model"])


class TestExpressionEncoder:
    def test_forward(self, config):
        cfg = config["model"]["expression_encoder"]
        model = ExpressionEncoder(
            input_dim=cfg["input_dim"], cnn_channels=cfg["cnn_channels"],
            cnn_kernel=cfg["cnn_kernel"], lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"], n_cancer_types=cfg["n_cancer_types"],
            cancer_embedding_dim=cfg["cancer_embedding_dim"],
            dropout=cfg["lstm_dropout"],
        )
        B = 4
        # Test the (B, F) input form (per-TEP feature vector)
        expr = torch.randn(B, 4)
        cancer = torch.randint(0, 33, (B,))
        out = model(expr, cancer)
        assert out.shape == (B, cfg["lstm_hidden"])


class TestTranslationEncoder:
    def test_forward(self, config):
        cfg = config["model"]["translation_encoder"]
        model = TranslationEncoder(
            input_dim=cfg["input_dim"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            n_inducing_points=cfg["n_inducing_points"], dropout=cfg["dropout"],
        )
        B = 4
        ribo = torch.randn(B, 10, 8)
        out = model(ribo)
        assert out.shape == (B, cfg["d_model"])


class TestTEFeatureEncoder:
    def test_forward(self, config):
        cfg = config["model"]["te_feature_encoder"]
        model = TEFeatureEncoder(
            input_dim=cfg["input_dim"], hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        )
        B = 8
        x = torch.randn(B, cfg["input_dim"])
        out = model(x)
        assert out.shape == (B, cfg["hidden_dims"][-1])


class TestGatedCrossAttentionFusion:
    def test_forward(self, config):
        cfg = config["model"]["fusion"]
        model = GatedCrossAttentionFusion(
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], dropout=cfg["dropout"],
            modality_dropout=cfg["modality_dropout"],
        )
        B = 4
        seq_emb = torch.randn(B, cfg["d_model"])
        expr_emb = torch.randn(B, cfg["d_model"])
        trans_emb = torch.randn(B, cfg["d_model"])
        te_feat_emb = torch.randn(B, cfg["d_model"])
        out = model(seq_emb, expr_emb, trans_emb, te_feat_emb)
        assert out.shape == (B, cfg["d_model"])


class TestDeepTEPeptideNet:
    def test_full_forward(self, config):
        model = DeepTEPeptideNet(config)
        B = 4
        batch = {
            "peptide_tokens": torch.randint(0, 26, (B, 25)),
            "mhc_tokens": torch.randint(0, 49, (B, 30)),
            "peptide_mask": torch.zeros(B, 25, dtype=torch.bool),
            "mhc_mask": torch.zeros(B, 30, dtype=torch.bool),
            "te_expression": torch.randn(B, 4),
            "ribo_features": torch.randn(B, 1, 8),
            "te_features": torch.randn(B, 64),
            "cancer_idx": torch.randint(0, 33, (B,)),
        }
        out = model(batch)
        assert "mhc_binding" in out
        assert "tcell_recognition" in out
        assert "immunogenicity_score" in out
        for _k, v in out.items():
            assert v.shape == (B,)
            assert (v >= 0).all() and (v <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
