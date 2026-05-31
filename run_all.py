#!/usr/bin/env python3
"""Projenin TEK kullanıcı giriş noktası.

``main.py`` pipeline mantığını korur; ``run_all.py`` onu *sarmalar* (wrap) ve
şunları ekler:

  * Ortam ön-kontrolü (venv, paketler, encoding, PYTHONPATH, veri dosyaları)
  * Aşama kısaltma çözümü     :  ``--stages 01 06`` → 01_dataset, 06_regression
  * Zincirleme garantisi      :  bir aşama, gereken önceki çıktılar yoksa durur
  * ``--resume``              :  son başarılı aşamadan devam (run_manifest.json)
  * ``--force``               :  cache yoksay, tüm seçili aşamaları yeniden üret
  * ``--quick``               :  küçük altküme / az iterasyon (hızlı duman testi)
  * Sade, renkli aşama banner'ları + bitişte özet rapor

Kullanım:
    python run_all.py                  # tüm pipeline
    python run_all.py --stages 01 06   # sadece bunlar
    python run_all.py --resume         # kaldığı yerden
    python run_all.py --force          # her şeyi yenile
    python run_all.py --quick          # hızlı test
"""

from __future__ import annotations

# ÖNEMLİ: Bu blok, ağır importlardan (torch, lightgbm, src.*) ÖNCE çalışmalı.
# Eksik paket/venv durumunda kullanıcı ImportError yerine net bir mesaj görür.
import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# ── TÜM uyarıları en başta sustur (ağır importlar ÖNCE bu env-var'ı görmeli) ──
# `python -W ignore` ile eşdeğer; child process'ler (joblib worker'ları,
# LightGBM/Torch C++ tarafı) da bu env'i miras alır.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")     # LightGBM C++ logger
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # (varsa) TF C++ logger
warnings.filterwarnings("ignore")                     # bu süreç için de
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent


def _silence_known_warnings() -> None:
    """Konsolu komple sustur: hiçbir 3.parti uyarı/gürültü görünmesin.

    Yalnızca bizim ``print``/log mesajlarımız ekranda kalsın. NumPy seterr,
    warnings filtresi ve gürültülü logger'lar (lightgbm, optuna, matplotlib,
    sklearn, py.warnings, numba, shap, urllib3, PIL) tamamen susturulur.
    """
    # 1) Python warnings — toptan kapat
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")
    # py.warnings logger'ı (logging.captureWarnings ile bağlanmış olabilir)
    logging.getLogger("py.warnings").setLevel(logging.ERROR)

    # 2) NumPy floating-point — sessiz; nan/inf'i np.where zaten yakalıyor
    try:
        import numpy as _np
        _np.seterr(divide="ignore", invalid="ignore", over="ignore", under="ignore")
    except Exception:
        pass

    # 3) Gürültülü 3.parti logger'ları ERROR'a çek
    for name in (
        "lightgbm", "optuna", "matplotlib", "PIL", "urllib3",
        "numba", "shap", "sklearn", "joblib", "torch", "tensorflow",
        "absl", "asyncio", "fsspec",
    ):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass

    # 4) Optuna'nın kendi log seviyesi (logging'den ayrı bir kanalı var)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Renkli/sade konsol yardımcıları (Windows terminalinde de çalışan ANSI)
# ----------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    """Metni ANSI renk koduyla sar (renk kapalıysa olduğu gibi döndür)."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _info(msg: str) -> None:
    print(_c(msg, "36"))          # camgöbeği


def _ok(msg: str) -> None:
    print(_c(msg, "32"))          # yeşil


def _warn(msg: str) -> None:
    print(_c(msg, "33"))          # sarı


def _err(msg: str) -> None:
    print(_c(msg, "31"))          # kırmızı


def _fmt_dur(sec: float) -> str:
    """Saniyeyi 'X dk Y sn' biçiminde okunur hale getir."""
    m, s = divmod(int(round(sec)), 60)
    return f"{m} dk {s} sn" if m else f"{s} sn"


# ----------------------------------------------------------------------------
# 1) ORTAM ÖN-KONTROLÜ  (yalnızca stdlib kullanır)
# ----------------------------------------------------------------------------
# import adı -> pip paket adı (mesajda doğru kurulum komutunu üretmek için)
_REQUIRED_PACKAGES: dict[str, str] = {
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "torch": "torch",
    "imblearn": "imbalanced-learn",
    "shap": "shap",
    "optuna": "optuna",
}


def _setup_encoding_and_path() -> None:
    """Windows için UTF-8 zorla ve proje kökünü PYTHONPATH'e ekle."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Mevcut sürecin stdout/stderr'ını da UTF-8'e çevir (Türkçe karakterler için)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    # `import src...` cwd'den bağımsız çalışsın diye kökü en başa koy
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _check_venv() -> list[str]:
    """Sanal ortam aktif mi? Değilse uyarı listesi döndür."""
    in_venv = (sys.prefix != getattr(sys, "base_prefix", sys.prefix)) or bool(
        os.environ.get("VIRTUAL_ENV")
    )
    if in_venv:
        return []
    return [
        "Sanal ortam (venv) aktif görünmüyor.",
        "  Aktifleştir:  .venv\\Scripts\\Activate.ps1   (Windows PowerShell)",
        "  veya çalıştır:  python run_all.py --skip-env-check  (riskine katlanarak)",
    ]


def _check_packages() -> list[str]:
    """Eksik paketleri bul; varsa pip install önerisiyle birlikte döndür."""
    missing = [
        pip_name
        for import_name, pip_name in _REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]
    if not missing:
        return []
    return [
        f"Eksik paket(ler): {', '.join(missing)}",
        f"  Kur:  pip install {' '.join(missing)}",
        "  veya:  pip install -e \".[dev]\"",
    ]


def _check_data() -> list[str]:
    """Zorunlu veri girdileri yerinde mi?"""
    problems: list[str] = []
    if not (ROOT / "data" / "raw").exists():
        problems.append("data/raw/ yok — harici dataset'i bağla:")
        problems.append('  python tools/import_dataset.py "C:\\path\\Dataset" --mode junction')
    gt = ROOT / "data" / "metadata" / "description-2.tab"
    if not gt.exists():
        problems.append("data/metadata/description-2.tab yok — ground truth dosyasını yerleştir.")
    return problems


def preflight(skip: bool) -> bool:
    """Tüm ortam kontrollerini çalıştır. Sorun varsa False döndür (dur)."""
    _setup_encoding_and_path()
    if skip:
        _warn("Ortam kontrolü atlandı (--skip-env-check).")
        return True

    problems: list[str] = []
    problems += _check_venv()
    problems += _check_packages()
    problems += _check_data()

    if problems:
        _err("Ortam kontrolü BAŞARISIZ:")
        for line in problems:
            _err("  " + line)
        return False
    _ok("Ortam kontrolü ✓ (venv, paketler, encoding, veri)")
    return True


# ----------------------------------------------------------------------------
# 2) AŞAMA KISALTMA ÇÖZÜMÜ
# ----------------------------------------------------------------------------
def resolve_stages(tokens: list[str], all_names: list[str]) -> tuple[list[str], list[str]]:
    """Kullanıcı token'larını ('01', '06') tam aşama adlarına eşle.

    Returns
    -------
    (resolved, unknown) : eşleşen tam adlar + eşleşmeyen token'lar
    """
    resolved: list[str] = []
    unknown: list[str] = []
    for tok in tokens:
        # Aşama adının alt-çizgiden önceki numara parçasıyla TAM eşleştir:
        #   "01"  → "01_dataset"      (01b/01c ile çakışmaz)
        #   "01b" → "01b_outliers"
        #   "06"  → "06_regression"
        # Tam ad da kabul edilir: "07_classification".
        matches = [n for n in all_names if n == tok or n.split("_", 1)[0] == tok]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            unknown.append(f"{tok} (belirsiz: {', '.join(matches)})")
        else:
            unknown.append(tok)
    return resolved, unknown


# ----------------------------------------------------------------------------
# 3) ZİNCİRLEME GARANTİSİ — 01_dataset çıktıları
# ----------------------------------------------------------------------------
# 01_dataset'in ürettiği ve sonraki neredeyse tüm aşamaların ihtiyaç duyduğu
# temel artefaktlar. Bunlar .npy (gitignore'da) → taze klonda yoktur, yeniden
# üretilmeleri gerekir.
_DATASET_FILES = ("X.npy", "y_chl.npy", "y_flav.npy", "y_nbi.npy", "y_stress.npy")


def _dataset_ready() -> bool:
    d = ROOT / "outputs" / "01_dataset"
    return all((d / f).exists() for f in _DATASET_FILES)


# ----------------------------------------------------------------------------
# 4) ÖZET RAPOR
# ----------------------------------------------------------------------------
def _tail_file(path: Path, max_lines: int = 30) -> str | None:
    """Bir metin dosyasının son satırlarını oku (özet ekranı için)."""
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def print_summary(results: list[dict], total_sec: float) -> None:
    """Bitişte sade özet: süre, aşama durumları, mevcut rapor özetleri."""
    print()
    print(_c("═" * 60, "36"))
    print(_c(f" ÖZET  ·  toplam süre: {_fmt_dur(total_sec)}", "36"))
    print(_c("═" * 60, "36"))
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    print(f" Aşama: {n_ok} ✓   {n_fail} ✗   {n_skip} atlandı")
    for r in results:
        mark = {"ok": "✓", "failed": "✗", "skipped": "—"}[r["status"]]
        col = {"ok": "32", "failed": "31", "skipped": "33"}[r["status"]]
        print("  " + _c(f"{mark} {r['name']:<24} {_fmt_dur(r['duration']):>10}", col))

    # Mevcut rapor artefaktlarının kuyruğunu bas (yeniden hesap yok)
    for label, rel in (
        ("Model özeti (14_model_summary/summary.txt)", "outputs/14_model_summary/summary.txt"),
        ("Ablation (15_ablation/ablation_report.md)", "outputs/15_ablation/ablation_report.md"),
    ):
        tail = _tail_file(ROOT / rel)
        if tail:
            print()
            print(_c(f"--- {label} ---", "36"))
            print(tail)


# ----------------------------------------------------------------------------
# 5) ANA AKIŞ
# ----------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    # Ortam hazır değilse hiçbir ağır import yapma, çık.
    if not preflight(args.skip_env_check):
        return 2

    # GÖREV 8: bilinen warning'leri konsoldan çıkar (ağır importlardan ÖNCE).
    _silence_known_warnings()

    # Ağır importlar yalnızca ortam doğrulandıktan sonra:
    from src.core import config, logging_setup, paths
    from main import build_pipeline

    cfg = config.load(args.config)
    if args.quick:
        cfg.set("quick", True)                     # alt modüller bu bayrağa bakar
        _warn("QUICK modu: küçük altküme / az iterasyon.")
    logging_setup.init(cfg)
    paths.ensure_outputs_tree()

    pipeline = build_pipeline(cfg, force=args.force)
    all_names = [name for name, _ in pipeline]

    # --- aşama seçimi ---
    if args.stages:
        selected, unknown = resolve_stages(args.stages, all_names)
        if unknown:
            _err("Tanınmayan aşama(lar): " + ", ".join(unknown))
            _info("Geçerli aşamalar: " + ", ".join(all_names))
            return 2
        selected_set = set(selected)
    else:
        selected_set = set(all_names)

    # --- resume: önceki manifest'te 'ok' olanları atla (force değilse) ---
    completed: set[str] = set()
    if args.resume and not args.force:
        mf = ROOT / "outputs" / "run_manifest.json"
        if mf.exists():
            try:
                prev = json.loads(mf.read_text(encoding="utf-8"))
                completed = {s["name"] for s in prev.get("stages", []) if s.get("status") == "ok"}
                _info(f"Resume: {len(completed)} tamamlanmış aşama atlanacak.")
            except Exception as exc:
                _warn(f"Önceki manifest okunamadı, baştan: {exc}")

    manifest = paths.start_manifest(cfg)
    results: list[dict] = []
    # Bu çalışmada 01_dataset üretilecek mi? (zincirleme kontrolü için)
    dataset_will_run = "01_dataset" in selected_set and "01_dataset" not in completed

    t_start = time.time()
    for name, fn in pipeline:
        if name not in selected_set or name in completed:
            results.append({"name": name, "status": "skipped", "duration": 0.0})
            continue

        # Zincirleme garantisi: dataset artefaktları yoksa ve bu çalışmada da
        # üretilmeyecekse, dur ve kullanıcıyı yönlendir.
        if name != "01_dataset" and not _dataset_ready() and not dataset_will_run:
            _err(f"'{name}' için 01_dataset çıktıları yok (X.npy, y_*.npy).")
            _err("Önce çalıştır:  python run_all.py --stages 01")
            manifest.record_stage(name, duration=0.0, status="failed")
            manifest.write()
            results.append({"name": name, "status": "failed", "duration": 0.0})
            break

        # --- aşama banner (başlangıç) ---
        print()
        print(_c("═" * 60, "36"))
        print(_c(f" {name}  başlıyor…", "36"))
        print(_c("═" * 60, "36"))

        t0 = time.time()
        try:
            fn()
            dur = time.time() - t0
            manifest.record_stage(name, duration=dur, status="ok")
            results.append({"name": name, "status": "ok", "duration": dur})
            print(_c("═" * 60, "32"))
            _ok(f" {name}  ({_fmt_dur(dur)})  ✓ TAMAMLANDI")
            print(_c("═" * 60, "32"))
        except Exception as exc:
            dur = time.time() - t0
            manifest.record_stage(name, duration=dur, status="failed")
            manifest.write()                       # kesilirse kaldığı yer kayıtlı
            results.append({"name": name, "status": "failed", "duration": dur})
            print(_c("═" * 60, "31"))
            _err(f" {name}  ({_fmt_dur(dur)})  ✗ HATA: {exc}")
            print(_c("═" * 60, "31"))
            _warn("Pipeline durdu. '--resume' ile bu aşamadan devam edebilirsin.")
            break

    manifest.write()
    print_summary(results, time.time() - t_start)
    # Çıkış kodu: herhangi bir başarısızlık varsa 1
    return 1 if any(r["status"] == "failed" for r in results) else 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_all.py",
        description="Hiperspektral pipeline — tek giriş noktası.",
    )
    p.add_argument("--config", default="config/default.yaml", help="YAML config yolu")
    p.add_argument("--stages", nargs="*", default=None,
                   help="Çalıştırılacak aşamalar (ör. 01 06 07_classification)")
    p.add_argument("--resume", action="store_true", help="Son başarılı aşamadan devam et")
    p.add_argument("--force", action="store_true", help="Cache yoksay, hepsini yenile")
    p.add_argument("--quick", action="store_true", help="Küçük altküme, hızlı test")
    p.add_argument("--skip-env-check", action="store_true", help="Ortam kontrolünü atla")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
