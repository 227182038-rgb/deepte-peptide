"""Training loop wrapping DeepTEPeptideNet with PyTorch Lightning-like interface."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from deepte.models.te_peptide_net import DeepTEPeptideNet
from deepte.training.losses import MultiTaskLoss
from deepte.utils.logging import get_logger

logger = get_logger(__name__)


def cosine_warmup_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr: float = 1e-6):
    """Cosine annealing with linear warmup."""

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        base_lr = optimizer.param_groups[0]["lr"]
        # Cosine decay relative to current LR
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        target_lr = min_lr + (base_lr - min_lr) * cosine_decay
        return target_lr / base_lr

    return LambdaLR(optimizer, lr_lambda)


class DeepTETrainer:
    """Custom training loop (avoids the overhead of PyTorch Lightning for full control)."""

    def __init__(
        self,
        config: dict,
        model: DeepTEPeptideNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device | None = None,
    ):
        self.cfg = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tcfg = config["training"]
        self.max_epochs = tcfg["max_epochs"]
        self.grad_clip = tcfg["gradient_clip_val"]
        self.accum_steps = tcfg["accumulate_grad_batches"]

        # Loss
        self.criterion = MultiTaskLoss(
            task_weights=tcfg["loss"]["task_weights"],
            pos_weight_tcell=tcfg["loss"]["pos_weight_tcell"],
            label_smoothing=tcfg["loss"]["label_smoothing"],
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=tcfg["optimizer"]["lr"],
            weight_decay=tcfg["optimizer"]["weight_decay"],
            betas=tuple(tcfg["optimizer"]["betas"]),
            eps=tcfg["optimizer"]["eps"],
        )

        # Scheduler
        self.scheduler = cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=tcfg["scheduler"]["warmup_steps"],
            max_steps=tcfg["scheduler"]["max_steps"],
            min_lr=tcfg["scheduler"]["min_lr"],
        )

        # Mixed precision (requires CUDA)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if (tcfg.get("precision") == "16-mixed" and torch.cuda.is_available())
            else None
        )

        # Early stopping
        self.patience = tcfg["early_stopping"]["patience"]
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Checkpoint dir
        self.ckpt_dir = Path(config["paths"]["model_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}

    def fit(self) -> dict:
        """Run full training. Returns final history."""
        self.model.to(self.device)
        global_step = 0

        for epoch in range(self.max_epochs):
            # ---------- TRAIN ----------
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.train_loader):
                batch = {
                    k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }

                if self.scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self.model(batch)
                        loss = self.criterion(outputs, batch)["loss"] / self.accum_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch)["loss"] / self.accum_steps
                    loss.backward()

                if (batch_idx + 1) % self.accum_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1

                epoch_loss += loss.item() * self.accum_steps
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)

            # ---------- VALIDATE ----------
            val_loss = self._validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Early stopping
            if val_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Save final model
        self._save_checkpoint(self.max_epochs - 1, self.best_val_loss, is_best=False, is_last=True)
        return self.history

    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {
                    k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }
                if self.scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self.model(batch)
                        loss = self.criterion(outputs, batch)["loss"]
                else:
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch)["loss"]
                total_loss += loss.item() * batch["peptide_tokens"].size(0)
                n += batch["peptide_tokens"].size(0)
        return total_loss / max(n, 1)

    def _save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False, is_last: bool = False
    ):
        ckpt = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.cfg,
        }
        if is_best:
            torch.save(ckpt, self.ckpt_dir / "best.ckpt")
            logger.info(f"Saved best checkpoint (val_loss={val_loss:.4f})")
        if is_last:
            torch.save(ckpt, self.ckpt_dir / "last.ckpt")
        torch.save(ckpt, self.ckpt_dir / f"epoch_{epoch+1:03d}.ckpt")
