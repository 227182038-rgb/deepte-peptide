"""Unit tests for predict CLI."""
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepte.cli.predict import predict_dataframe
from deepte.models.te_peptide_net import DeepTEPeptideNet


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
                "lstm_dropout": 0.0, "n_cancer_types": 10,
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


def test_predict_dataframe(mini_config):
    df = pd.DataFrame({
        "peptide_seq": ["MAAAAA", "MLTLLPTTGA"],
        "hla_allele": ["HLA-A*02:01", "HLA-A*01:01"],
        "cancer_type": ["BRCA", "LUAD"],
        "te_expression": [5.0, 4.0],
        "riboseq_psite_count": [10, 5],
        "riboseq_periodicity": [0.6, 0.4],
    })
    model = DeepTEPeptideNet(mini_config)
    res = predict_dataframe(df, model, mini_config, device=torch.device("cpu"))
    assert "pred_mhc_binding" in res.columns
    assert "pred_tcell_recognition" in res.columns
    assert "pred_immunogenicity_score" in res.columns
    assert len(res) == 2
    assert (res["pred_immunogenicity_score"] >= 0).all()
    assert (res["pred_immunogenicity_score"] <= 1).all()
