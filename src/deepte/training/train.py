"""Main training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deepte.models.te_peptide_net import DeepTEPeptideNet
from deepte.training.trainer import DeepTETrainer
from deepte.utils.io import load_config, set_seed
from deepte.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TEPDataset(Dataset):
    """PyTorch dataset for candidate TEPs.

    Reads a features.npz file produced by deepte.features.build_features.
    """

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=True)
        self.sequence = data["sequence"]
        self.mhc_allele = data["mhc_allele"]
        self.expression = data["expression"]
        self.translation = data["translation"]
        self.te_features = data["te_features"]
        self.cancer_idx = data["cancer_idx"]
        self.label_mhc = data["label_mhc_binding"]
        self.label_tcell = data["label_tcell_recognition"]
        self.label_immuno = data["label_immunogenicity"]

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, idx: int):
        return {
            "peptide_tokens": torch.from_numpy(self.sequence[idx]).long(),
            "mhc_tokens": torch.from_numpy(
                self.mhc_allele[idx].argmax(axis=-1)
            ).long(),  # onehot -> idx
            "peptide_mask": (self.sequence[idx] == 0),
            "mhc_mask": (self.mhc_allele[idx].sum(axis=-1) == 0),
            "te_expression": torch.from_numpy(self.expression[idx]).float(),
            "ribo_features": torch.from_numpy(self.translation[idx])
            .float()
            .unsqueeze(0),  # (N=1, 8)
            "te_features": torch.from_numpy(self.te_features[idx]).float(),
            "cancer_idx": torch.from_numpy(np.array(self.cancer_idx[idx])).long(),
            "mhc_binding": torch.from_numpy(np.array(self.label_mhc[idx])).float(),
            "tcell_recognition": torch.from_numpy(np.array(self.label_tcell[idx])).float(),
            "immunogenicity_score": torch.from_numpy(np.array(self.label_immuno[idx])).float(),
        }


# ---------------------------------------------------------------------------
# K-fold split
# ---------------------------------------------------------------------------
def make_cancer_type_stratified_split(
    metadata_path: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Cancer-type-stratified split."""
    import pandas as pd

    meta = pd.read_csv(metadata_path, sep="\t")
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for _cancer, group in meta.groupby("cancer_type"):
        idxs = group.index.tolist()
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepTE-Peptide")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--features",
        default="data/processed/features.npz",
        help="Path to features.npz produced by build_features.py",
    )
    parser.add_argument("--metadata", default="data/processed/metadata.tsv")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold (0..n-1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config and set seed
    cfg = load_config(args.config)
    set_seed(args.seed, deterministic=cfg.get("deterministic", True))
    setup_logging(
        level="INFO", log_file=Path(cfg["paths"]["log_dir"]) / f"train_fold{args.fold}.log"
    )

    # Load data
    logger.info(f"Loading features from {args.features}")
    dataset = TEPDataset(args.features)
    logger.info(f"Dataset size: {len(dataset):,}")

    train_idx, val_idx, _ = make_cancer_type_stratified_split(
        args.metadata,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        seed=args.seed,
    )

    # Subset to the current fold
    n_folds = cfg["cross_validation"]["n_folds"]
    fold_size = len(train_idx) // n_folds
    start = args.fold * fold_size
    end = (args.fold + 1) * fold_size if args.fold < n_folds - 1 else len(train_idx)
    val_idx = train_idx[start:end]
    train_idx = train_idx[:start] + train_idx[end:]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        drop_last=cfg["training"]["drop_last"],
        persistent_workers=cfg["training"]["persistent_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # Build model
    model = DeepTEPeptideNet(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Train
    trainer = DeepTETrainer(cfg, model, train_loader, val_loader)
    history = trainer.fit()

    # Save history
    from deepte.utils.io import save_json

    history_path = Path(cfg["paths"]["report_dir"]) / f"train_history_fold{args.fold}.json"
    save_json(history, history_path)
    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
