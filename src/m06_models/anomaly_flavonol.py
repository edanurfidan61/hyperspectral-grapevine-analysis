"""Flavonol kalite kontrolü için anomali tespiti.

"Normal" tanım: y_stress == 0 (sağlıklı) ve y_flav >= 3.5 (Ph.Eur. PASS).
Dedektörler yalnız normal örneklerle eğitilir; sonra 204 yaprağın tamamında
değerlendirilir. Anomaly = y_flav < 3.5.

Üç dedektör:
    - IsolationForest    (sklearn)
    - OneClassSVM        (sklearn)
    - Autoencoder        (PyTorch MLP: input → 32 → 8 → 32 → input)

Çıktılar:
    - outputs/10_anomaly_flavonol/anomaly_flavonol_report.csv
    - outputs/10_anomaly_flavonol/anomaly_flavonol_plot.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.core import paths
from src.core.logging_setup import get as get_logger

log = get_logger("m06_models.anomaly_flavonol")

PHEUR_THRESHOLD = 3.5


def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds_dir = paths.OUTPUTS_DIR / "01_dataset"
    X = np.load(ds_dir / "X.npy")
    y_flav = np.load(ds_dir / "y_flav.npy")
    y_stress = np.load(ds_dir / "y_stress.npy")
    return X, y_flav, y_stress


def _train_autoencoder(X_train: np.ndarray, input_dim: int, random_state: int):
    """3 katmanlı MLP autoencoder (input → 32 → 8 → 32 → input).

    Küçük dataset için: dropout + weight decay + early stopping (val split).
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    class AE(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 8), nn.ReLU(),
            )
            self.dec = nn.Sequential(
                nn.Linear(8, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, dim),
            )

        def forward(self, x):
            return self.dec(self.enc(x))

    n = X_train.shape[0]
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    val_size = max(2, n // 5)  # ~%20 val
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    X_tr = torch.from_numpy(X_train[tr_idx].astype(np.float32))
    X_val = torch.from_numpy(X_train[val_idx].astype(np.float32))

    model = AE(input_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = min(8, max(2, X_tr.shape[0]))
    max_epochs = 300
    patience = 25
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0

    for epoch in range(max_epochs):
        model.train()
        perm_t = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], batch_size):
            xb = X_tr[perm_t[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val), X_val).item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if (epoch + 1) % 25 == 0:
            log.info("  AE epoch %d val=%.5f (best=%.5f, bad=%d)",
                     epoch + 1, val_loss, best_val, bad)
        if bad >= patience:
            log.info("  AE early stopping @ epoch %d (best val=%.5f)", epoch + 1, best_val)
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _best_threshold(scores: np.ndarray, y_anomaly: np.ndarray) -> float:
    """Youden-J: ROC üzerinde TPR-FPR'ı maksimize eden eşik."""
    if len(np.unique(y_anomaly)) < 2:
        return float(np.median(scores))
    fpr, tpr, thr = roc_curve(y_anomaly, scores)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def _autoencoder_recon_error(model, X: np.ndarray) -> np.ndarray:
    import torch

    X_t = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        Xh = model(X_t).cpu().numpy()
    return np.mean((X - Xh) ** 2, axis=1)


def _eval_scores(y_anomaly: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> dict:
    """Anomali skoru ile tahmin etiketleri üzerinden metrikler.

    AUC skorun yönünden bağımsız raporlanır: AUC<0.5 ise skor ters çevrilir.
    Bu, "anomali ne tarafta" konvansiyonunu kalibre etmek yerine algoritmanın
    ayırt etme gücünü ölçmek için yapılır (one-class dedektörlerde skor yönü
    bazen ters olabilir).
    """
    prec = float(precision_score(y_anomaly, labels, zero_division=0))
    rec = float(recall_score(y_anomaly, labels, zero_division=0))
    f1 = float(f1_score(y_anomaly, labels, zero_division=0))
    try:
        auc = float(roc_auc_score(y_anomaly, scores))
        if auc < 0.5:
            auc = 1.0 - auc
    except ValueError:
        auc = float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


def _plot_recon_distribution(
    errors: np.ndarray,
    y_anomaly: np.ndarray,
    save_path: Path,
    threshold: float | None = None,
) -> None:
    """Recon-error histogramı: log-x ekseni + 99-yüzdelik kırpma + opsiyonel eşik çizgisi."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    normal_err = errors[y_anomaly == 0]
    anomaly_err = errors[y_anomaly == 1]

    # log-x: 0 ve negatifleri küçük pozitif değere çek
    eps = max(1e-6, float(np.min(errors[errors > 0])) * 0.5) if np.any(errors > 0) else 1e-6
    n_safe = np.maximum(normal_err, eps)
    a_safe = np.maximum(anomaly_err, eps)

    upper = float(np.percentile(errors, 99))
    bins = np.logspace(np.log10(eps), np.log10(max(upper, eps * 10)), 30)

    ax.hist(n_safe, bins=bins, alpha=0.6,
            label=f"Normal (PASS, n={len(normal_err)})",
            color="steelblue", edgecolor="k")
    ax.hist(a_safe, bins=bins, alpha=0.6,
            label=f"Anomaly (FAIL, n={len(anomaly_err)})",
            color="indianred", edgecolor="k")
    if threshold is not None and threshold > 0:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
                   label=f"Eşik (Youden-J) = {threshold:.4g}")
    ax.set_xscale("log")
    ax.set_xlabel("Autoencoder reconstruction error (MSE, log)")
    ax.set_ylabel("Frekans")
    ax.set_title("Flavonol anomali — yeniden yapılandırma hatası dağılımı",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(cfg=None) -> Path:
    """Pipeline aşaması: flavonol anomali tespiti."""
    random_state = int(cfg.get("models.random_state", 42)) if cfg is not None else 42
    threshold = float(cfg.get("targets.pheur_flavonol_threshold", PHEUR_THRESHOLD)) if cfg is not None else PHEUR_THRESHOLD

    X, y_flav, y_stress = _load_data()
    valid = np.isfinite(y_flav)
    Xv, yv_flav, yv_stress = X[valid], y_flav[valid], y_stress[valid]

    normal_mask = (yv_stress == 0) & (yv_flav >= threshold)
    y_anomaly = (yv_flav < threshold).astype(int)
    log.info("Anomaly setup: total=%d, normal_train=%d, anomaly=%d",
             len(yv_flav), int(normal_mask.sum()), int(y_anomaly.sum()))

    if int(normal_mask.sum()) < 5:
        raise RuntimeError(
            f"Yeterli normal örnek yok (n={int(normal_mask.sum())}); "
            f"y_stress==0 ve y_flav>={threshold} koşulunu sağlayan en az 5 yaprak gerekir."
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(Xv[normal_mask])
    X_all = scaler.transform(Xv)

    rows: list[dict] = []

    # ---- IsolationForest ----------------------------------------------------
    log.info("IsolationForest eğitiliyor...")
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_all)  # büyük = daha anomali
    iso_pred_raw = iso.predict(X_all)
    iso_pred = (iso_pred_raw == -1).astype(int)
    iso_metrics = _eval_scores(y_anomaly, iso_scores, iso_pred)
    rows.append({"detector": "isolation_forest", **iso_metrics})
    log.info("IF: P=%.3f R=%.3f F1=%.3f AUC=%.3f",
             iso_metrics["precision"], iso_metrics["recall"],
             iso_metrics["f1"], iso_metrics["auc_roc"])

    # ---- OneClassSVM --------------------------------------------------------
    log.info("OneClassSVM eğitiliyor...")
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    ocsvm.fit(X_train)
    ocsvm_scores = -ocsvm.score_samples(X_all)
    ocsvm_pred_raw = ocsvm.predict(X_all)
    ocsvm_pred = (ocsvm_pred_raw == -1).astype(int)
    ocsvm_metrics = _eval_scores(y_anomaly, ocsvm_scores, ocsvm_pred)
    rows.append({"detector": "one_class_svm", **ocsvm_metrics})
    log.info("OCSVM: P=%.3f R=%.3f F1=%.3f AUC=%.3f",
             ocsvm_metrics["precision"], ocsvm_metrics["recall"],
             ocsvm_metrics["f1"], ocsvm_metrics["auc_roc"])

    # ---- Autoencoder --------------------------------------------------------
    log.info("Autoencoder eğitiliyor (PyTorch)...")
    ae_threshold = None
    try:
        ae = _train_autoencoder(X_train, input_dim=X_train.shape[1], random_state=random_state)
        recon_err = _autoencoder_recon_error(ae, X_all)

        # Eşik seçimi: normal örneklerin 95p'i (etiket sızıntısız) — referans
        train_err = _autoencoder_recon_error(ae, X_train)
        thr_p95 = float(np.percentile(train_err, 95))
        # Youden-J optimal eşik (data-driven; flav etiketinden faydalanır)
        thr_youden = _best_threshold(recon_err, y_anomaly)
        ae_threshold = thr_youden
        ae_pred = (recon_err > ae_threshold).astype(int)
        ae_metrics = _eval_scores(y_anomaly, recon_err, ae_pred)
        rows.append({"detector": "autoencoder", **ae_metrics,
                     "threshold": ae_threshold, "threshold_p95_train": thr_p95})
        log.info("AE: P=%.3f R=%.3f F1=%.3f AUC=%.3f thr_youden=%.5g (p95_train=%.5g)",
                 ae_metrics["precision"], ae_metrics["recall"],
                 ae_metrics["f1"], ae_metrics["auc_roc"], thr_youden, thr_p95)
    except Exception as exc:
        log.exception("Autoencoder eğitilemedi: %s", exc)
        recon_err = None

    # ---- Çıktıları yaz ------------------------------------------------------
    # Tek kaynak: numaralı aşama dizini (eski outputs/reports/ yazımı kaldırıldı).
    stage_dir = paths.stage_dir("10_anomaly_flavonol")
    out_csv = stage_dir / "anomaly_flavonol_report.csv"
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(out_csv, index=False, encoding="utf-8")
    log.info("Rapor yazıldı: %s", out_csv)

    if recon_err is not None:
        plot_path = stage_dir / "anomaly_flavonol_plot.png"
        _plot_recon_distribution(recon_err, y_anomaly, plot_path, threshold=ae_threshold)
        log.info("Grafik yazıldı: %s", plot_path)

    paths.write_source_marker(
        stage_dir,
        producer="src/m06_models/anomaly_flavonol.py",
        config_source=cfg.source if cfg is not None else None,
    )
    return out_csv


if __name__ == "__main__":
    run()
