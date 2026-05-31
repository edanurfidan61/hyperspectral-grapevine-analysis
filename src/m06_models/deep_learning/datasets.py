"""Numpy → PyTorch Dataset/DataLoader yardımcıları + DL veri hazırlığı.

- ``NumpyDataset``: opsiyonel augmentation (Gaussian gürültü + spektral kaydırma)
- ``prepare_dl_data``: NaN filtreleme, scaling, stratified split (tek-split)
- ``prepare_dl_kfolds``: K-fold CV üreteci (her fold ölçekli + stratified)
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from src.core.logging_setup import get as get_logger

log = get_logger("m06_models.deep_learning.datasets")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class NumpyDataset(Dataset):
        def __init__(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            *,
            augment_noise: float = 0.0,
            augment_shift: int = 0,
        ):
            self.X = np.asarray(X, dtype=np.float32)
            self.augment_noise = float(augment_noise)
            self.augment_shift = int(augment_shift)
            if y is None:
                self.y = None
                self._y_int = False
                return
            y_arr = np.asarray(y)
            if y_arr.ndim >= 2 or np.issubdtype(y_arr.dtype, np.floating):
                self.y = y_arr.astype(np.float32)
                self._y_int = False
            elif np.issubdtype(y_arr.dtype, np.integer):
                self.y = y_arr.astype(np.int64)
                self._y_int = True
            else:
                self.y = y_arr.astype(np.float32)
                self._y_int = False

        def __len__(self) -> int:
            return int(self.X.shape[0])

        def _augment(self, x: np.ndarray) -> np.ndarray:
            if self.augment_noise > 0.0:
                x = x + np.random.normal(0.0, self.augment_noise, size=x.shape).astype(np.float32)
            if self.augment_shift > 0:
                shift = int(np.random.randint(-self.augment_shift, self.augment_shift + 1))
                if shift != 0:
                    x = np.roll(x, shift, axis=-1)
            return x

        def __getitem__(self, idx: int):
            x = self.X[idx]
            if self.augment_noise > 0.0 or self.augment_shift > 0:
                x = self._augment(x.copy())
            x_t = torch.as_tensor(x, dtype=torch.float32)
            if self.y is None:
                return x_t, torch.as_tensor(0.0, dtype=torch.float32)
            y_item = self.y[idx]
            if self._y_int:
                return x_t, torch.as_tensor(int(y_item), dtype=torch.long)
            return x_t, torch.as_tensor(np.asarray(y_item, dtype=np.float32))

    def make_loader(
        X: np.ndarray,
        y: Optional[np.ndarray],
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        augment_kwargs: Optional[dict] = None,
    ):
        ds = NumpyDataset(X, y, **(augment_kwargs or {}))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

else:
    class NumpyDataset:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available — use sklearn fallback")

    def make_loader(*args, **kwargs):  # type: ignore
        raise RuntimeError("PyTorch not available — use sklearn fallback")


def _split_clean(X: np.ndarray, y: np.ndarray, regression: bool):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if regression:
        valid = np.isfinite(y)
    else:
        valid = np.isfinite(np.asarray(y, dtype=np.float64))
    Xv, yv = X[valid], y[valid]
    if not regression:
        yv = yv.astype(np.int64)
    return Xv, yv, valid


def prepare_dl_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    regression: bool,
    val_size: float = 0.2,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, "object"]:
    """Tek-split: NaN at, ölçekle, stratified ya da group split."""
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    Xv, yv, valid_mask = _split_clean(X, y, regression)
    gv = groups[valid_mask] if groups is not None else None
    if Xv.shape[0] < 4:
        raise RuntimeError(f"Eğitim için yeterli örnek yok: n={Xv.shape[0]}")

    if gv is not None:
        # Group split — aynı yaprak hep aynı tarafta
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        tr_idx, va_idx = next(gss.split(Xv, yv, groups=gv))
        X_train, X_val = Xv[tr_idx], Xv[va_idx]
        y_train, y_val = yv[tr_idx], yv[va_idx]
    else:
        stratify = None
        if not regression:
            unique, counts = np.unique(yv, return_counts=True)
            if len(unique) > 1 and counts.min() >= 2:
                stratify = yv
            else:
                log.warning("Stratify atlandı: sınıf dağılımı %s",
                            dict(zip(unique.tolist(), counts.tolist())))

        X_train, X_val, y_train, y_val = train_test_split(
            Xv, yv, test_size=val_size, random_state=random_state, stratify=stratify
        )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    if regression:
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
    return X_train, X_val, y_train, y_val, scaler


def prepare_dl_kfolds(
    X: np.ndarray,
    y: np.ndarray,
    *,
    regression: bool,
    n_splits: int = 5,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, "object"]]:
    """K-fold üreteci: her fold için (X_tr, X_va, y_tr, y_va, scaler).

    Sınıflandırmada en küçük sınıf ``n_splits``'ten büyükse stratified, değilse
    standart KFold. ``groups`` verilirse Group(Stratified)KFold ile leakage önlenir.
    """
    from sklearn.preprocessing import StandardScaler

    from src.core.cv import make_cv_splitter, split_indices

    Xv, yv, valid_mask = _split_clean(X, y, regression)
    gv = groups[valid_mask] if groups is not None else None
    n = Xv.shape[0]
    if n < n_splits * 2:
        log.warning("n=%d için n_splits=%d büyük; tek-split'e düşülüyor", n, n_splits)
        Xt, Xva, yt, yva, sc = prepare_dl_data(
            X, y, regression=regression, val_size=1.0 / max(n_splits, 2),
            random_state=random_state, groups=groups,
        )
        yield Xt, Xva, yt, yva, sc
        return

    task = "regression" if regression else "classification"
    try:
        kf = make_cv_splitter(
            n_splits=n_splits, task=task, groups=gv, random_state=random_state,
        )
        splits = list(split_indices(kf, Xv, yv, gv))
    except ValueError as exc:
        log.warning("Stratify/group splitter başarısız (%s); KFold'a düşüldü", exc)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(Xv))

    for tr_idx, va_idx in splits:
        X_train, X_val = Xv[tr_idx], Xv[va_idx]
        y_train, y_val = yv[tr_idx], yv[va_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        if regression:
            y_train = y_train.astype(np.float32)
            y_val = y_val.astype(np.float32)
        yield X_train, X_val, y_train, y_val, scaler
