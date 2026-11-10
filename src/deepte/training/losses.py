"""Multi-task loss with class imbalance handling and label smoothing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Combined loss for the three DeepTE-Peptide output heads.

    Total loss = w_mhc * BCE(mhc_binding) + w_tcell * Focal(tcell_recognition)
        + w_immuno * MSE(immunogenicity)
    """

    def __init__(
        self,
        task_weights: dict[str, float] | None = None,
        pos_weight_tcell: float = 5.0,
        label_smoothing: float = 0.05,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.task_weights = task_weights or {
            "mhc_binding": 1.0,
            "tcell_recognition": 1.5,
            "immunogenicity_score": 1.2,
        }
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma

        # BCE for mhc_binding with positive weight for imbalance
        self.bce_mhc = nn.BCELoss()

        # Focal loss for T-cell recognition (heavily imbalanced)
        self.register_buffer(
            "pos_weight_tcell",
            torch.tensor(pos_weight_tcell, dtype=torch.float32),
        )

        # MSE for immunogenicity regression
        self.mse = nn.MSELoss()

    def focal_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary focal loss with class weights."""
        # Smooth labels
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy(preds, targets, reduction="none")
        p_t = torch.where(targets > 0.5, preds, 1 - preds)
        focal_weight = (1 - p_t) ** self.focal_gamma
        # Apply positive class weight
        pos_mask = (targets > 0.5).float()
        weight = pos_mask * self.pos_weight_tcell + (1 - pos_mask) * 1.0
        return (focal_weight * weight * bce).mean()

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute the combined multi-task loss.

        Returns
        -------
        Dict[str, torch.Tensor]
            {"loss": total, "loss_mhc": ..., "loss_tcell": ..., "loss_immuno": ...}
        """
        loss_mhc = self.bce_mhc(outputs["mhc_binding"], targets["mhc_binding"].float())
        loss_tcell = self.focal_loss(
            outputs["tcell_recognition"], targets["tcell_recognition"].float()
        )
        loss_immuno = self.mse(
            outputs["immunogenicity_score"], targets["immunogenicity_score"].float()
        )

        total = (
            self.task_weights["mhc_binding"] * loss_mhc
            + self.task_weights["tcell_recognition"] * loss_tcell
            + self.task_weights["immunogenicity_score"] * loss_immuno
        )

        return {
            "loss": total,
            "loss_mhc": loss_mhc.detach(),
            "loss_tcell": loss_tcell.detach(),
            "loss_immuno": loss_immuno.detach(),
        }
