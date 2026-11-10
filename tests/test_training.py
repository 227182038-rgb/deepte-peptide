"""Unit tests for training losses and trainer."""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepte.models.te_peptide_net import DeepTEPeptideNet
from deepte.training.losses import MultiTaskLoss
from deepte.training.trainer import DeepTETrainer


class TestMultiTaskLoss:
    def test_loss_computation(self):
        loss_fn = MultiTaskLoss()
        outputs = {
            "mhc_binding": torch.tensor([0.2, 0.8, 0.5, 0.9]),
            "tcell_recognition": torch.tensor([0.1, 0.7, 0.3, 0.85]),
            "immunogenicity_score": torch.tensor([0.15, 0.75, 0.4, 0.88]),
        }
        targets = {
            "mhc_binding": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "tcell_recognition": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "immunogenicity_score": torch.tensor([0.1, 0.8, 0.3, 0.9]),
        }
        losses = loss_fn(outputs, targets)
        assert "loss" in losses
        assert "loss_mhc" in losses
        assert "loss_tcell" in losses
        assert "loss_immuno" in losses
        assert losses["loss"].item() > 0.0
        assert torch.isfinite(losses["loss"])

    def test_focal_loss_weighting(self):
        loss_fn = MultiTaskLoss(pos_weight_tcell=10.0, label_smoothing=0.0)
        # Prediction is wrong on positive sample vs wrong on negative sample
        pred_wrong_pos = torch.tensor([0.1])
        target_pos = torch.tensor([1.0])
        l_pos = loss_fn.focal_loss(pred_wrong_pos, target_pos)

        pred_wrong_neg = torch.tensor([0.9])
        target_neg = torch.tensor([0.0])
        l_neg = loss_fn.focal_loss(pred_wrong_neg, target_neg)

        # Due to positive weighting (10.0 vs 1.0), positive error should yield higher loss
        assert l_pos > l_neg


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, size=8):
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
            "mhc_binding": torch.tensor(1.0 if idx % 2 == 0 else 0.0),
            "tcell_recognition": torch.tensor(1.0 if idx % 2 == 0 else 0.0),
            "immunogenicity_score": torch.tensor(0.8 if idx % 2 == 0 else 0.1),
        }


class TestDeepTETrainer:
    def test_trainer_fit_and_checkpoint(self, tmp_path):
        cfg = {
            "training": {
                "max_epochs": 2,
                "batch_size": 4,
                "eval_batch_size": 4,
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 1,
                "precision": "32-true",
                "early_stopping": {"patience": 2},
                "optimizer": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
                "scheduler": {
                    "warmup_steps": 2,
                    "max_steps": 10,
                    "min_lr": 1e-5,
                },
                "loss": {
                    "task_weights": {
                        "mhc_binding": 1.0,
                        "tcell_recognition": 1.0,
                        "immunogenicity_score": 1.0,
                    },
                    "pos_weight_tcell": 2.0,
                    "label_smoothing": 0.0,
                },
            },
            "model": {
                "sequence_encoder": {
                    "vocab_size": 26, "d_model": 32, "n_heads": 2, "n_layers": 1,
                    "dropout": 0.0, "max_peptide_len": 25, "max_mhc_allele_len": 30,
                },
                "expression_encoder": {
                    "input_dim": 1, "cnn_channels": [8, 16], "cnn_kernel": 3,
                    "lstm_hidden": 16, "lstm_layers": 1, "cancer_embedding_dim": 8,
                    "lstm_dropout": 0.0, "n_cancer_types": 10,
                },
                "translation_encoder": {
                    "input_dim": 8, "d_model": 16, "n_heads": 2, "n_layers": 1,
                    "n_inducing_points": 4, "dropout": 0.0,
                },
                "te_feature_encoder": {
                    "input_dim": 64, "hidden_dims": [32, 16], "dropout": 0.0,
                    "activation": "gelu", "batch_norm": False,
                },
                "fusion": {
                    "d_model": 32, "n_heads": 2, "n_layers": 1,
                    "dropout": 0.0, "modality_dropout": 0.0,
                },
                "heads": {
                    "mhc_binding": {"hidden_dims": [16], "output_dim": 1},
                    "tcell_recognition": {"hidden_dims": [16], "output_dim": 1},
                    "immunogenicity_score": {"hidden_dims": [16], "output_dim": 1},
                },
            },
            "paths": {
                "model_dir": str(tmp_path / "models"),
            },
        }

        model = DeepTEPeptideNet(cfg)
        train_ds = DictDataset(8)
        val_ds = DictDataset(4)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        trainer = DeepTETrainer(
            cfg, model, train_loader, val_loader, device=torch.device("cpu")
        )
        history = trainer.fit()

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert (tmp_path / "models" / "best.ckpt").exists()
        assert (tmp_path / "models" / "last.ckpt").exists()
