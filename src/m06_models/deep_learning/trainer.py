"""PyTorch eğitim döngüsü: early stopping + checkpoint + sağlam loss şekillendirme.

`Trainer` PyTorch yoksa `RuntimeError` atar — modeller sklearn fallback'e düşer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

from src.core.logging_setup import get as get_logger

log = get_logger("m06_models.deep_learning.trainer")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def _align_for_loss(out, yb, criterion):
    """Loss çağrısından önce şekilleri uyumlandır.

    - CrossEntropy: ``out`` (B, C), ``yb`` (B,) long
    - MSE/L1: skaler hedef için trailing singleton boyutlarını sıkıştır
    - AE rekonstrüksiyon: ``out`` ve ``yb`` aynı 2D şekilde geçer
    """
    if isinstance(criterion, nn.CrossEntropyLoss):
        return out, yb.long()

    # Regresyon: hedef 1D ise out'u (B,) yap
    if yb.dim() == 1:
        while out.dim() > 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out, yb.float()

    # AE: ikisi de 2D — direkt geç
    return out, yb.float()


if TORCH_AVAILABLE:
    class Trainer:
        def __init__(
            self,
            model: nn.Module,
            device: Optional[str] = None,
            epochs: int = 100,
            batch_size: int = 32,
            lr: float = 1e-3,
            patience: int = 15,
            weight_decay: float = 0.0,
            amp: bool = False,
        ) -> None:
            self.model = model
            if isinstance(device, str) and device.lower() == "cuda" and not torch.cuda.is_available():
                log.warning("CUDA istenmiş ama yok; CPU'ya geçiliyor")
                device = "cpu"
            self.device = (
                torch.device(device)
                if isinstance(device, str)
                else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            )
            self.epochs = int(epochs)
            self.batch_size = int(batch_size)
            self.lr = float(lr)
            self.patience = int(patience)
            self.weight_decay = float(weight_decay)
            self.amp = bool(amp) and self.device.type == "cuda"
            self.model.to(self.device)

        def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            criterion: Optional[nn.Module] = None,
            checkpoint_path: Optional[Path] = None,
        ) -> Dict[str, Any]:
            if optimizer is None:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            if criterion is None:
                criterion = nn.MSELoss()

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=max(3, self.patience // 3)
            )

            best_val = float("inf")
            no_improve = 0
            history = {"train_loss": [], "val_loss": []}

            scaler = torch.cuda.amp.GradScaler() if self.amp else None

            for epoch in range(1, self.epochs + 1):
                self.model.train()
                train_losses = []
                for xb, yb in train_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    optimizer.zero_grad()
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            out = self.model(xb)
                            out_a, yb_a = _align_for_loss(out, yb, criterion)
                            loss = criterion(out_a, yb_a)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out = self.model(xb)
                        out_a, yb_a = _align_for_loss(out, yb, criterion)
                        loss = criterion(out_a, yb_a)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                    train_losses.append(float(loss.detach().cpu().item()))

                avg_train = float(sum(train_losses) / max(1, len(train_losses)))
                history["train_loss"].append(avg_train)

                val_loss = None
                if val_loader is not None and len(val_loader) > 0:
                    self.model.eval()
                    vals = []
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb = xb.to(self.device)
                            yb = yb.to(self.device)
                            out = self.model(xb)
                            out_a, yb_a = _align_for_loss(out, yb, criterion)
                            vals.append(float(criterion(out_a, yb_a).detach().cpu().item()))
                    val_loss = float(sum(vals) / max(1, len(vals)))
                    history["val_loss"].append(val_loss)
                    scheduler.step(val_loss)
                    if val_loss + 1e-9 < best_val:
                        best_val = val_loss
                        no_improve = 0
                        if checkpoint_path is not None:
                            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save({"model_state": self.model.state_dict()}, checkpoint_path)
                    else:
                        no_improve += 1

                log.info("Epoch %d/%d — train=%.4f val=%s", epoch, self.epochs, avg_train,
                         f"{val_loss:.4f}" if val_loss is not None else "(na)")
                if val_loader is not None and no_improve >= self.patience:
                    log.info("Early stopping (patience=%d, best_val=%.4f)", self.patience, best_val)
                    break

            # En iyi ağırlıkları geri yükle
            if checkpoint_path is not None and Path(checkpoint_path).exists():
                try:
                    state = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(state["model_state"])
                except Exception as exc:
                    log.warning("Checkpoint geri yüklenemedi: %s", exc)

            return history

        def predict(self, loader: DataLoader):
            import numpy as np

            self.model.eval()
            out_list = []
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(self.device)
                    out = self.model(xb)
                    out_list.append(out.detach().cpu().numpy())
            if not out_list:
                return np.empty((0,))
            arr = np.concatenate(out_list, axis=0)
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr.reshape(arr.shape[0])
            return arr

else:
    class Trainer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available — Trainer requires torch.")
