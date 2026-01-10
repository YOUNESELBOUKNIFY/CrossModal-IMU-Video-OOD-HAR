"""
Trainer classes pour pretraining et classification

✅ Fix importants (par rapport à ton ancien code):
- Gestion robuste warmup=0 (LinearLR crash sinon)
- support vidéo (B,T,C,H,W) vs (B,C,T,H,W) via config.data.video_channel_first
- best_metric cohérent:
    - pretrain => minimise val_loss
    - classification => maximise balanced_accuracy
- checkpoints complets (optimizer + scheduler)
- sauvegarde "last.pt" + "best_model.pt"
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm


# -----------------------------
#   Base Trainer
# -----------------------------
class BaseTrainer:
    def __init__(self, model, config, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.current_epoch = 0
        self.history = {"train": [], "val": []}

    def save_checkpoint(self, path: Path, extra: Optional[Dict[str, Any]] = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, str(path))
        print(f"[Checkpoint] saved -> {path}")

    def load_checkpoint(self, path: Path, strict: bool = True) -> Dict[str, Any]:
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        self.current_epoch = int(ckpt.get("epoch", 0))
        self.history = ckpt.get("history", {"train": [], "val": []})
        print(f"[Checkpoint] loaded <- {path}")
        return ckpt


# -----------------------------
#   Cross-modal Pretraining Trainer
# -----------------------------
class CrossModalTrainer(BaseTrainer):
    """
    Pretraining cross-modal IMU-Video
    best = min val_loss
    """

    def __init__(self, model, loss_fn, config, device: str = "cuda"):
        super().__init__(model, config, device)
        self.loss_fn = loss_fn

        self.best_val_loss = float("inf")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.pretrain_lr,
            weight_decay=config.training.pretrain_weight_decay,
        )

        # Scheduler: warmup + cosine
        num_epochs = int(config.training.pretrain_epochs)
        warmup_epochs = int(getattr(config.training, "pretrain_warmup_epochs", 0))

        if warmup_epochs <= 0:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(num_epochs, 1),
                eta_min=1e-6,
            )
        else:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(num_epochs - warmup_epochs, 1),
                eta_min=1e-6,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )

        # Vidéo format
        self.video_channel_first = bool(getattr(config.data, "video_channel_first", False))

    def _maybe_permute_video(self, video: torch.Tensor) -> torch.Tensor:
        # Dataset peut retourner (T,C,H,W) ou (C,T,H,W)
        # Batch => (B, ...)
        # Si tu veux (B,C,T,H,W), mets config.data.video_channel_first=True
        if self.video_channel_first:
            # Attendu (B,C,T,H,W). Si entrée (B,T,C,H,W) -> permute
            if video.dim() == 5 and video.shape[1] != 3 and video.shape[2] == 3:
                video = video.permute(0, 2, 1, 3, 4)
        else:
            # Attendu (B,T,C,H,W). Si entrée (B,C,T,H,W) -> permute
            if video.dim() == 5 and video.shape[1] == 3:
                video = video.permute(0, 2, 1, 3, 4)
        return video

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"[Pretrain] Epoch {self.current_epoch}", leave=False)

        for batch in pbar:
            imu = batch["imu"].to(self.device, non_blocking=True)      # (B,C,T)
            video = batch["video"].to(self.device, non_blocking=True)  # (B,T,C,H,W) or (B,C,T,H,W)
            video = self._maybe_permute_video(video)

            imu_proj, video_proj = self.model(imu, video)
            loss = self.loss_fn(imu_proj, video_proj)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        return total_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def validate(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="[Pretrain] Val", leave=False):
            imu = batch["imu"].to(self.device, non_blocking=True)
            video = batch["video"].to(self.device, non_blocking=True)
            video = self._maybe_permute_video(video)

            imu_proj, video_proj = self.model(imu, video)
            loss = self.loss_fn(imu_proj, video_proj)
            total_loss += float(loss.item())

        return total_loss / max(len(dataloader), 1)

    def fit(self, train_loader, val_loader):
        num_epochs = int(self.config.training.pretrain_epochs)
        save_dir = Path(self.config.paths.checkpoints_dir) / "cross_modal"
        save_dir.mkdir(parents=True, exist_ok=True)

        patience = int(getattr(self.config.training, "patience", 10))
        save_every = int(getattr(self.config.training, "save_every", 5))
        save_best_only = bool(getattr(self.config.training, "save_best_only", True))

        patience_counter = 0

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)

            self.scheduler.step()

            print(f"[Pretrain] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            # Save last
            self.save_checkpoint(
                save_dir / "last.pt",
                extra={
                    "best_val_loss": self.best_val_loss,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                },
            )

            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                if save_best_only:
                    self.save_checkpoint(
                        save_dir / "best_model.pt",
                        extra={
                            "best_val_loss": self.best_val_loss,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                        },
                    )
            else:
                patience_counter += 1

            # Periodic
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    save_dir / f"checkpoint_epoch_{epoch}.pt",
                    extra={
                        "best_val_loss": self.best_val_loss,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                    },
                )

            if patience_counter >= patience:
                print(f"[Pretrain] Early stopping at epoch {epoch}")
                break

        with open(save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)


# -----------------------------
#   Classification Trainer
# -----------------------------
class ClassificationTrainer(BaseTrainer):
    """
    Downstream classification (IMU only)
    - mode='linear_probe': freeze encoder, train head
    - mode='finetune'    : unfreeze encoder, train all
    best = max balanced_accuracy
    """

    def __init__(self, model, config, device: str = "cuda", mode: str = "linear_probe"):
        super().__init__(model, config, device)
        assert mode in ["linear_probe", "finetune"]
        self.mode = mode

        self.loss_fn = nn.CrossEntropyLoss()

        # Best metric: maximize balanced accuracy
        self.best_bal_acc = 0.0

        # Freeze / unfreeze
        if mode == "linear_probe":
            for p in model.imu_encoder.parameters():
                p.requires_grad = False

            self.optimizer = AdamW(
                model.classifier.parameters(),
                lr=config.training.train_lr_head,
                weight_decay=config.training.pretrain_weight_decay,
            )

        else:  # finetune
            # Si ta classe n'a pas unfreeze_encoder, fallback ici
            if hasattr(model, "unfreeze_encoder"):
                model.unfreeze_encoder()
            else:
                for p in model.imu_encoder.parameters():
                    p.requires_grad = True

            self.optimizer = AdamW(
                [
                    {"params": model.imu_encoder.parameters(), "lr": config.training.train_lr_encoder},
                    {"params": model.classifier.parameters(), "lr": config.training.train_lr_head},
                ],
                weight_decay=config.training.pretrain_weight_decay,
            )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(int(config.training.train_epochs), 1),
            eta_min=1e-7,
        )

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"[Cls:{self.mode}] Epoch {self.current_epoch}", leave=False)

        for batch in pbar:
            imu = batch["imu"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            logits = self.model(imu)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            total += labels.size(0)
            correct += int((pred == labels).sum().item())

            pbar.set_postfix(loss=float(loss.item()), acc=100.0 * correct / max(total, 1))

        return {"loss": total_loss / max(len(dataloader), 1), "accuracy": 100.0 * correct / max(total, 1)}

    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"[Cls:{self.mode}] Val", leave=False):
            imu = batch["imu"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            logits = self.model(imu)
            loss = self.loss_fn(logits, labels)

            total_loss += float(loss.item())
            pred = logits.argmax(dim=1)

            total += labels.size(0)
            correct += int((pred == labels).sum().item())

            all_preds.extend(pred.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        from sklearn.metrics import balanced_accuracy_score, f1_score

        acc = 100.0 * correct / max(total, 1)
        bal_acc = 100.0 * balanced_accuracy_score(all_labels, all_preds)
        f1 = 100.0 * f1_score(all_labels, all_preds, average="macro")

        return {
            "loss": total_loss / max(len(dataloader), 1),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_macro": f1,
        }

    def fit(self, train_loader, val_loader) -> float:
        num_epochs = int(self.config.training.train_epochs)
        save_dir = Path(self.config.paths.checkpoints_dir) / f"classifier_{self.mode}"
        save_dir.mkdir(parents=True, exist_ok=True)

        patience = int(getattr(self.config.training, "patience", 10))
        patience_counter = 0

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            self.scheduler.step()

            print(
                f"[Cls:{self.mode}] epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.2f}% | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.2f}% "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.2f}% val_f1={val_metrics['f1_macro']:.2f}%"
            )

            # Save last
            self.save_checkpoint(
                save_dir / "last.pt",
                extra={
                    "best_balanced_accuracy": self.best_bal_acc,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                },
            )

            # Save best (maximize balanced accuracy)
            if val_metrics["balanced_accuracy"] > self.best_bal_acc:
                self.best_bal_acc = float(val_metrics["balanced_accuracy"])
                patience_counter = 0
                self.save_checkpoint(
                    save_dir / "best_model.pt",
                    extra={
                        "best_balanced_accuracy": self.best_bal_acc,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                    },
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"[Cls:{self.mode}] Early stopping at epoch {epoch}")
                break

        with open(save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.best_bal_acc
