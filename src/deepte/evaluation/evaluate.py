"""Evaluator: runs inference on test sets, computes metrics, saves reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepte.evaluation.metrics import bootstrap_ci, compute_all_metrics
from deepte.models.te_peptide_net import DeepTEPeptideNet
from deepte.training.train import TEPDataset
from deepte.utils.io import load_config
from deepte.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class Evaluator:
    """Run inference, compute metrics, and save evaluation reports."""

    def __init__(self, config: dict, model_ckpt: Path):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepTEPeptideNet(config)
        ckpt = torch.load(model_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> dict[str, np.ndarray]:
        """Run inference on a dataloader and return aggregated predictions."""
        preds = {k: [] for k in ["mhc_binding", "tcell_recognition", "immunogenicity_score"]}
        targets = {k: [] for k in preds}
        for batch in loader:
            x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = self.model(x)
            for k in preds:
                preds[k].append(out[k].cpu().numpy())
                targets[k].append(batch[k].cpu().numpy())
        return {
            "predictions": {k: np.concatenate(v) for k, v in preds.items()},
            "targets": {k: np.concatenate(v) for k, v in targets.items()},
        }

    def evaluate(self, loader: DataLoader) -> dict:
        """Compute all metrics for each output head."""
        result = self.predict(loader)
        metrics = {}
        for head in ["mhc_binding", "tcell_recognition", "immunogenicity_score"]:
            y_true = result["targets"][head]
            y_score = result["predictions"][head]
            # Binarise for classification metrics (positive label = 1)
            y_true_bin = (y_true > 0.5).astype(int)
            head_metrics = compute_all_metrics(y_true_bin, y_score)
            # Bootstrap CI for AUC-ROC
            _, ci_low, ci_high = bootstrap_ci(
                y_true_bin,
                y_score,
                lambda y, s: __import__(
                    "sklearn.metrics", fromlist=["roc_auc_score"]
                ).roc_auc_score(y, s),
                n_bootstrap=500,
            )
            head_metrics["auc_roc_ci_low"] = ci_low
            head_metrics["auc_roc_ci_high"] = ci_high
            metrics[head] = head_metrics
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepTE-Peptide")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/reports/evaluation.json")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(level="INFO")

    dataset = TEPDataset(args.features)
    # Use test split (last 15%)
    from deepte.training.train import make_cancer_type_stratified_split

    _, _, test_idx = make_cancer_type_stratified_split(
        args.metadata,
        train_frac=0.7,
        val_frac=0.15,
        seed=42,
    )
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    evaluator = Evaluator(cfg, Path(args.checkpoint))
    metrics = evaluator.evaluate(loader)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation to {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
