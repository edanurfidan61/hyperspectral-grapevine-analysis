# Hyperspectral Grapevine Leaf Analysis — Disease Detection & Flavonol Prediction

> Hiperspektral asma yaprağı analizi — hastalık tespiti & ilaç hammaddesi (flavonol) tahmini

Machine-learning pipeline that predicts **biochemical content** (chlorophyll,
flavonol, NBI) and classifies **stress / disease state** (4 classes, incl.
*flavescence dorée*) of grapevine (*Vitis vinifera*) leaves directly from
**hyperspectral images** (204 bands, 397–1004 nm).

**Biomedical motivation:** Grapevine (red vine) leaf is a herbal drug raw
material; leaf flavonoids are the active ingredient of venous-circulation drugs
such as **Antistax®** (AS 195), which is based on the French Pharmacopoeia
"Vigne Rouge" monograph and is authorised via the well-established-use (WEU) /
traditional-use (TU) route — there is **no Ph.Eur. monograph**. The EMA
assessment report (EMA/HMPC/464682/2016) reports a flavonoid level of ≈3.5% for
red vine leaf, which we use as an **operational PASS/FAIL threshold**. This
project links three questions end-to-end: *detect the disease → measure its
effect on flavonol → decide pass/fail (EMA ≈3.5% threshold)* from a single
hyperspectral scan.

---

<details>
<summary><b>🇹🇷 Türkçe — Proje Özeti (genişletmek için tıklayın)</b></summary>

Asma yaprağının (*Vitis vinifera*) hiperspektral görüntüsünden (204 bant,
397–1004 nm) hem **biyokimyasal içeriğini** (klorofil, flavonol, NBI) tahmin
eden hem de **stres/hastalık durumunu** (4 sınıf — sağlıklı, flavescence dorée,
diğer biyotik, abiyotik) sınıflandıran uçtan uca bir makine öğrenmesi
pipeline'ı.

**Biyomedikal motivasyon:** Kırmızı asma yaprağı bir bitkisel ilaç
hammaddesidir; **Antistax®** (AS 195) gibi venöz dolaşım ilaçlarının etken
maddesi yaprak flavonoidleridir. Antistax, Fransız Farmakopesi "Vigne Rouge"
monografına dayanır ve iyi-yerleşmiş kullanım (WEU) / geleneksel kullanım (TU)
üzerinden ruhsatlıdır; **Ph.Eur. monografı yoktur**. EMA değerlendirme raporu
(EMA/HMPC/464682/2016) kırmızı asma yaprağı için ≈%3.5 flavonoid düzeyi
bildirir; bunu **operasyonel PASS/FAIL eşiği** olarak kullanıyoruz. Proje
üç soruyu birleştirir: *hastalığı tespit et → flavonole etkisini ölç → EMA
≈%3.5 eşiğiyle geçti/kaldı kararı ver.*

Ayrıntılı Türkçe raporlar: [`proje_ozeti.md`](proje_ozeti.md) (algoritma/akış
odaklı) ve [`SONUC_OZETI.md`](SONUC_OZETI.md) (sonuç/tablo odaklı).

</details>

---

## 🔑 Key Results / Ana Sonuçlar

| Task / Görev | Best model | Score | Note |
|---|---|---|---|
| Chlorophyll regression | PLS (tuned) | **R² = 0.855** | Excellent (RPD 2.62) |
| NBI regression | PLS (tuned) | **R² = 0.764** | Good (RPD 2.06) |
| Flavonol regression | LightGBM (tuned) | **R² ≈ 0.39** | Hard (weak UV-blue signal) |
| Stress classification (4-class) | LightGBM (tuned) | **Acc 0.80 / Macro-F1 0.77** | |
| **Flavescence dorée (FD) detection** | LightGBM | **Recall 0.89 / F1 0.85** | Disease is clearly detectable |
| Pass/Fail (EMA ≈3.5% threshold) | SVM | **Acc 0.76** | FAIL recall 0.87 |

**🔬 Headline finding:** Flavescence dorée disease does **not lower** flavonol —
it **raises** it (FD mean 3.51 vs healthy 3.27; %56 pass vs %30). Flavonoids are
defence compounds; the plant up-regulates them under stress. Diseased leaves can
paradoxically be **more valuable** as raw material.

**⚠️ Methodological caveat:** The genetic-algorithm band selection appears to
reach R² ≈ 0.64 for flavonol, but **nested cross-validation reveals this is
selection bias** — the honest value is **≈ 0.30**. GA's real value is *wavelength
interpretability*, not an R² boost. See §"GA" in [`proje_ozeti.md`](proje_ozeti.md).

---

## 🚀 Quick Start / Hızlı Başlangıç

```bash
# 1. Install / Kurulum
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -e ".[dev]"

# 2. Run the whole pipeline / Tüm pipeline'ı çalıştır
python run_all.py
```

`run_all.py` is the single entry point: it checks the environment (venv,
packages, encoding, data), runs all 23 stages in order, and prints a summary.

```bash
python run_all.py --stages 01 06   # only these stages (short numbers accepted)
python run_all.py --resume         # continue from last successful stage
python run_all.py --force          # ignore cache, rebuild all
python run_all.py --quick          # small subset, fast smoke test
```

---

## 📦 Data / Veri

This repo ships the **processed dataset** so you can reproduce all modelling
without the 42 GB of raw ENVI cubes. The raw dataset contains **205 leaves**;
one is dropped because its segmentation mask comes back empty, so all modelling
runs on **204 leaves**.

| Included in repo | Path | Size |
|---|---|---|
| Processed feature matrix | `outputs/01_dataset/X.npy` (204×520) | ~0.8 MB |
| Targets | `outputs/01_dataset/y_{chl,flav,nbi,stress}.npy` | <0.1 MB |
| Feature names / groups | `outputs/01_dataset/*.json`, `groups_*.npy` | <0.1 MB |
| Ground truth | `data/metadata/description-2.tab` | ~18 KB |

> **Raw hyperspectral cubes (42 GB) are NOT in the repo.** They live under
> `data/raw/` locally and are git-ignored. To regenerate the feature matrix from
> raw cubes, link your dataset and run stage 01:
>
> ```bash
> python tools/import_dataset.py "C:\path\to\Dataset" --mode junction
> python run_all.py --stages 01
> ```
>
> Source dataset: Ryckewaert et al. grapevine hyperspectral leaf dataset.

---

## 🧠 Algorithms & How They Connect / Algoritmalar ve Bağlantıları

The pipeline is a **network, not a straight line.** Three algorithm families:

```
                 ┌──────────────────────────────────────────────┐
                 │  01_dataset (X, y) — source for ALL stages    │
                 └───────────────┬──────────────────────────────┘
        ┌────────────────────────┼──────────────────────────────┐
   [Feature selection]      [Modelling]                   [Validation]
   04 SHAP                  06 Regression ─┐               01c Holdout (KL-seed)
   05 RFE        ─► which   07 Classify    ├─► 06b/07b      15 Ablation
   05b PLS-VIP      bands?  08 Deep Learn  │   Tuning:      16 Final + nested-CV
   12 GA        ─┘          09 Ordinal     │   RandSearch
        │                   10 Anomaly     │   → Optuna
        └─► 12b consensus   11 Ensemble    │
            (wavelengths)        │         └─► 14 Model Summary (merges all)
                            13/13b Flavonol combos
```

1. **Feature selection (SHAP, RFE, PLS-VIP, GA):** four independent methods
   answer "which of the 520 features carry signal?" — they cross-validate each
   other; their consensus (12b/13b) yields interpretable wavelengths.
2. **Predictors (PLS, Ridge, RF, LightGBM, Stacking, 7 DL nets):** compete on the
   same data; two-stage tuning = **RandomizedSearchCV (coarse) → Optuna (fine)**.
3. **Validation layer (group-CV, holdout, ablation, nested-CV):** guarantees the
   numbers are not inflated — the methodological backbone (e.g. nested-CV exposes
   GA's selection bias).

Full per-stage explanation of *what each algorithm does and how it links to the
others*: see [`proje_ozeti.md`](proje_ozeti.md).

---

## 📁 Output Structure / Çıktı Yapısı

Outputs are organised by **stage number** under `outputs/`:

```
outputs/
├── 01_dataset/                 X.npy, y_*.npy, groups_*.npy   (✓ in repo)
├── 01b_outliers/  01c_holdout/ outlier report, train/test split + seed search
├── 02_eda/  03_visualization/  EDA; per-leaf maps (03 git-ignored, 530 MB)
├── 04_feature_shap/ 05_feature_rfe/ 05b_pls_vip/   feature importance
├── 06_regression/ 06b_regression_tuned/            6 regressors + tuning
├── 07_classification(_resampled)/ 07b_..._tuned/   4-class + flavonoid PASS/FAIL binary (EMA ≈3.5%)
├── 08_deep_learning/           7 architectures
├── 09_ordinal_flavonol/ 10_anomaly_flavonol/ 11_ensemble/
├── 12_ga_feature_selection/ 12b_..._consensus/     GA bands + wavelengths
├── 13_flavonol_combos/ 13b_feature_consensus/
├── 14_model_summary/           all_models.csv (consolidated, 101 rows)
├── 15_ablation/                component-contribution analysis
└── 16_final_combos/            F1–F5 + nested-CV (selection-bias proof)
```

Each folder has a `source.txt` (which module produced it, with which config).

---

## 🗺️ Module Map / Modül Haritası

| Folder | Responsibility |
|---|---|
| `src/core/` | Config, paths, logging, CV splitters, spectral utils |
| `src/m01_io/` | ENVI cube reading |
| `src/m02_preprocessing/` | Spectral correction (SG, SNV, MSC, derivative) + segmentation |
| `src/m03_indices/` | 13 vegetation indices (NDVI, ARI, FLAVI…) |
| `src/m04_features/` | Feature vector + SHAP / RFE / GA / PLS-VIP selection |
| `src/m05_dataset/` | Leaf loop, ground-truth mapping, X/y build, visualisation |
| `src/m06_models/` | BaseModel + regressors + classifiers + 7 DL nets + tuning + ablation |
| `src/m07_ensemble/` | Ensemble strategies + consolidated model summary |

> Python package names can't start with a digit, hence the `m01_`, `m02_` …
> prefixes under `src/`; `outputs/` folders use bare numbers for readability.

---

## ⚙️ Configuration / Konfigürasyon

All hyperparameters, wavelengths, thresholds, CV/seed live in
[`config/default.yaml`](config/default.yaml). For experiments, copy it and run
`python main.py --config config/your_exp.yaml`.

---

## 🧪 Tests & Dev

```bash
pytest tests/
black src/ tools/ tests/
ruff check src/ tools/ tests/
```

`tools/` holds single-purpose helpers: `import_dataset.py` (link external
dataset) and `run_single_model.py` (isolated model debug).

---

## 🔧 Troubleshooting / Sorun Giderme

| Error | Cause | Fix |
|---|---|---|
| `Sanal ortam (venv) aktif görünmüyor` | venv not active | `.venv\Scripts\Activate.ps1`, or `--skip-env-check` |
| `Eksik paket(ler): ...` | deps missing | `pip install -e ".[dev]"` |
| `'06_regression' için 01_dataset çıktıları yok` | `X.npy` missing | `python run_all.py --stages 01` first |

---

## 📊 Reports / Raporlar

- [`proje_ozeti.md`](proje_ozeti.md) — algorithm & data-flow walkthrough (TR)
- [`SONUC_OZETI.md`](SONUC_OZETI.md) — comprehensive results & tables (TR)
- `outputs/14_model_summary/all_models.csv` — every model in one table

---

*Pipeline: 23 stages, single command (`python run_all.py`), group-aware CV
(leakage-free), CPU-only. Last full run: 2026-05-31 (FD class included).*
