"""Unit tests for ensemble predictions."""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepte.models.ensemble import EnsemblePredictor
from deepte.models.te_peptide_net import DeepTEPeptideNet


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, size=6):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "peptide_tokens": torch.randint(0, 26, (25,)),
            "mhc_tokens": torch.randint(0, 49, (30,)),
            "peptide_mask": torch.zeros(25, dtype=torch.bool),
            "mhc_mask": torch.zeros(30, dtype=torch.bool),
            "te_expression": torch.randn(4),
            "ribo_features": torch.randn(1, 8),
            "te_features": torch.randn(64),
            "cancer_idx": torch.tensor(0, dtype=torch.long),
        }


@pytest.fixture
def mini_config():
    return {
        "model": {
            "sequence_encoder": {
                "vocab_size": 26, "d_model": 16, "n_heads": 2, "n_layers": 1,
                "dropout": 0.0, "max_peptide_len": 25, "max_mhc_allele_len": 30,
            },
            "expression_encoder": {
                "input_dim": 1, "cnn_channels": [4, 8], "cnn_kernel": 3,
                "lstm_hidden": 8, "lstm_layers": 1, "cancer_embedding_dim": 4,
                "lstm_dropout": 0.0, "n_cancer_types": 5,
            },
            "translation_encoder": {
                "input_dim": 8, "d_model": 8, "n_heads": 2, "n_layers": 1,
                "n_inducing_points": 2, "dropout": 0.0,
            },
            "te_feature_encoder": {
                "input_dim": 64, "hidden_dims": [16, 8], "dropout": 0.0,
                "activation": "gelu", "batch_norm": False,
            },
            "fusion": {
                "d_model": 16, "n_heads": 2, "n_layers": 1,
                "dropout": 0.0, "modality_dropout": 0.0,
            },
            "heads": {
                "mhc_binding": {"hidden_dims": [8], "output_dim": 1},
                "tcell_recognition": {"hidden_dims": [8], "output_dim": 1},
                "immunogenicity_score": {"hidden_dims": [8], "output_dim": 1},
            },
        }
    }


def test_ensemble_aggregation(tmp_path, mini_config):
    # Save 2 dummy model checkpoints
    ckpt_paths = []
    for i in range(2):
        model = DeepTEPeptideNet(mini_config)
        p = tmp_path / f"model_{i}.pt"
        torch.save({"state_dict": model.state_dict()}, p)
        ckpt_paths.append(p)

    loader = DataLoader(DictDataset(6), batch_size=3)

    # Test mean aggregation
    ens_mean = EnsemblePredictor(mini_config, ckpt_paths, aggregation="mean")
    preds = ens_mean.predict(loader)
    assert "mhc_binding" in preds
    assert len(preds["mhc_binding"]) == 6

    # Test uncertainty
    unc = ens_mean.predict_uncertainty(loader)
    assert "mean" in unc["mhc_binding"]
    assert "std" in unc["mhc_binding"]
    assert len(unc["mhc_binding"]["mean"]) == 6
