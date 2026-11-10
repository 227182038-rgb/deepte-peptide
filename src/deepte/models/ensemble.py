"""Ensemble of DeepTE-Peptide models for cross-validation inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepte.models.te_peptide_net import DeepTEPeptideNet
from deepte.utils.logging import get_logger

logger = get_logger(__name__)


class EnsemblePredictor:
    """Aggregates predictions from multiple DeepTE-Peptide folds."""

    def __init__(self, config: dict, model_paths: list[Path], aggregation: str = "mean"):
        self.config = config
        self.models: list[DeepTEPeptideNet] = []
        self.aggregation = aggregation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for path in model_paths:
            model = DeepTEPeptideNet(config)
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()
            model.to(self.device)
            self.models.append(model)
            logger.info(f"Loaded model from {path}")

    def predict(self, dataloader: DataLoader) -> dict[str, np.ndarray]:
        """Run inference across all models in the ensemble.

        Returns
        -------
        Dict[str, np.ndarray]
            Aggregated predictions per output head.
        """
        all_preds: dict[str, list[np.ndarray]] = {
            k: [] for k in ["mhc_binding", "tcell_recognition", "immunogenicity_score"]
        }

        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }
                fold_preds = {k: [] for k in all_preds}
                for model in self.models:
                    out = model(batch)
                    for k in all_preds:
                        fold_preds[k].append(out[k].cpu().numpy())
                # Aggregate across folds for this batch
                for k in all_preds:
                    stacked = np.stack(fold_preds[k], axis=0)  # (n_folds, B)
                    if self.aggregation == "mean":
                        agg = stacked.mean(axis=0)
                    elif self.aggregation == "median":
                        agg = np.median(stacked, axis=0)
                    elif self.aggregation == "rank_average":
                        from scipy.stats import rankdata

                        ranks = np.array([rankdata(p) for p in stacked])
                        agg = ranks.mean(axis=0) / len(stacked[0])
                    else:
                        agg = stacked.mean(axis=0)
                    all_preds[k].append(agg)

        return {k: np.concatenate(v) for k, v in all_preds.items()}

    def predict_uncertainty(self, dataloader: DataLoader) -> dict[str, dict[str, np.ndarray]]:
        """Return per-sample mean and standard deviation across folds."""
        all_preds: dict[str, list[np.ndarray]] = {
            k: [] for k in ["mhc_binding", "tcell_recognition", "immunogenicity_score"]
        }

        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }
                fold_preds = {k: [] for k in all_preds}
                for model in self.models:
                    out = model(batch)
                    for k in all_preds:
                        fold_preds[k].append(out[k].cpu().numpy())
                for k in all_preds:
                    all_preds[k].append(np.stack(fold_preds[k], axis=0))

        result = {}
        for k, v in all_preds.items():
            stacked = np.concatenate(v, axis=1)  # (n_folds, N)
            result[k] = {
                "mean": stacked.mean(axis=0),
                "std": stacked.std(axis=0),
                "min": stacked.min(axis=0),
                "max": stacked.max(axis=0),
            }
        return result
