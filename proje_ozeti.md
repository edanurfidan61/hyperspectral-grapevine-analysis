# Hiperspektral Asma Yaprağı Analizi — Pipeline, Algoritmalar ve Bağlantıları

> Bu rapor, projenin **23 pipeline aşamasını**, kullanılan **algoritmaların ne işe
> yaradığını, nasıl kullanıldığını ve birbirleriyle nasıl bağlandığını** açıklar.
> Hedef okuyucu: hiperspektral ML konusuna yabancı biri (örn. jüri). Tüm sayılar
> 2026-05-31 tarihli son tam çalıştırmadan (FD sınıfı dahil, türev bloğu canlı)
> alınmıştır. Sonuç-odaklı tablo derlemesi için ayrıca `SONUC_OZETI.md`'ye bakınız.

---

## Giriş — Proje ne yapıyor?

Proje, **asma yaprağının (*Vitis vinifera*) hiperspektral görüntüsünden** hem
**biyokimyasal içeriğini** (klorofil, flavonol, NBI) tahmin eden hem de
**stres/hastalık durumunu** sınıflandıran uçtan uca bir makine öğrenmesi
pipeline'ıdır.

**Biyomedikal motivasyon:** 
 Asma yaprağı, geleneksel bitkisel tıbbi ürünlerde kullanılan bir ilaç hammaddesidir; Antistax® gibi kronik venöz yetmezlik ilaçlarının etken maddesi yaprak flavonoidleridir. EMA değerlendirme raporu (EMA/HMPC/464682/2016), kırmızı asma yaprağında flavonoid içeriğini en fazla %3.5 olarak bildirir; bu çalışmada söz konusu değer, hammadde uygunluğu için operasyonel bir PASS/FAIL eşiği olarak benimsenmiştir. Ayrıca flavescence dorée (FD), asmanın ekonomik açıdan en yıkıcı fitoplazma hastalığıdır. Proje bu iki ekseni birleştirir: hastalığı tespit et → hastalığın ilaç hammaddesine etkisini ölç → "geçti/kaldı" kararı ver.

**Veri:** 205 ham yaprak görüntüsü; 1'i segmentasyon maskesi boş döndüğü için elenir → 204 ile çalışılır. 204 spektral bant (397–1004 nm, VNIR), ENVI küp.
**Temel kısıt:** Küçük örneklem (n=204) + yüksek boyut → aşırı uydurma (overfitting)
riski tüm tasarım kararlarının merkezindedir.

---

## Algoritmalar arası genel akış (kuş bakışı)

Pipeline doğrusal bir zincir değil, **birbirini besleyen bir ağdır.** Ana bağlantılar:

```
                 ┌─────────────────────────────────────────────┐
                 │  01_dataset (X, y_*) — TÜM aşamaların kaynağı │
                 └───────────────┬─────────────────────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────────┐
        │                        │                              │
   [Keşif/seçim]           [Modelleme]                    [Doğrulama]
   02 EDA                  06 Regresyon  ─┐                01c Holdout
   03 Görsel               07 Sınıflama   ├─► 06b/07b      15 Ablation
   04 SHAP  ──┐            08 Derin Öğr.  │   (Tuning:      16 Final+nested
   05 RFE     ├─► hangi    09 Ordinal     │   RandSearch
   05b VIP    │   bantlar  10 Anomali     │   + Optuna)
   12 GA  ────┘   önemli?  11 Ensemble    │
        │                        │        │
        └──► 12b konsensüs       └────────┴──► 14 Model Özeti (hepsini birleştirir)
             (dalga boyları)          │
                                  13/13b Flavonol kombinasyonları
```

**Üç algoritma ailesi ve görev paylaşımı:**

1. **Öznitelik seçimi/önem (SHAP, RFE, PLS-VIP, GA):** "204 banttan + 13 indeksten
   hangileri gerçekten bilgi taşıyor?" sorusunu farklı açılardan yanıtlar. Çıktıları
   hem yorumlanabilirlik (hangi dalga boyu) hem de boyut indirgeme sağlar.
2. **Tahmin modelleri (PLS, Ridge, RF, LightGBM, Stacking, DL):** Asıl regresyon ve
   sınıflandırmayı yapar. Aynı veri üzerinde yarıştırılır.
3. **Doğrulama/dürüstlük katmanı (grup-CV, holdout, ablation, nested-CV):** Sonuçların
   şişmediğini garanti eder — bu projenin metodolojik omurgası.

---

## Aşama 01 — Dataset Oluşturma (`src/m05_dataset/builder.py`)

Ham hiperspektral küp doğrudan modele verilmez; **biyofiziksel olarak anlamlı 520
öznitelikli** bir vektöre dönüştürülür. Bu, küçük veride ham 204×H×W pikselle
çalışmaktan çok daha sağlamdır.

**Öznitelik blokları (toplam 520):**

| Blok | Sayı | Nasıl üretilir | Hangi algoritma kullanır |
|---|---|---|---|
| SNV spektrum | 204 | Standard Normal Variate ile normalize tam spektrum | Tüm modeller, GA, SHAP |
| 1. türev (SG-d1) | 204 | Savitzky-Golay birinci türev (pencere 11, derece 2) | Flavonol için en bilgili blok (bkz. Ablation B4) |
| İndeks + sabit | 112 | 13 vejetasyon indeksi (NDVI, ARI, ZTM, SIPI, FLAVI…) + NBI haritaları + continuum-removal + kenar özellikleri | Stres sınıflama, F5 |

**Ön işleme algoritmalarının rolü:**
- **SNV:** Aydınlatma/saçılma farklarını giderir → spektrumları karşılaştırılabilir kılar.
- **Savitzky-Golay 1. türev:** Taban çizgisi kaymasını yok eder, absorpsiyon
  kenarlarını (özellikle flavonol için kritik) vurgular.
- **Vejetasyon indeksleri:** Bilinen biyokimyasal oranları (örn. FLAVI flavonole,
  ZTM klorofile duyarlı) hazır öznitelik olarak ekler.

**Çıktı:** `X.npy` (204×520), `y_chl/flav/nbi/stress.npy`, `groups_leaf/plot.npy`.
Bu çıktı **sonraki 22 aşamanın tamamının girdisidir.**

> **Metodolojik not (düzeltilmiş bug):** 1. türev bloğunun 204 kolonu bir ara
> çalıştırmada yanlışlıkla sıfırlanıyordu (ablation aşaması veri setini "savgol
> kapalı" haliyle yeniden kurup eski haline döndürmüyordu). Düzeltildi; türev bloğu
> artık canlı ve ablation veri setini her zaman geri kuruyor.

---

## Aşama 01b–01c — Aykırı Değer Filtresi ve Holdout Seçimi

- **01b (`outlier_filter.py`):** IQR tabanlı aykırı değer işaretleme (action=warn —
  silmez, raporlar). Küçük veride örnek atmamak için muhafazakâr.
- **01c (`holdout_split.py`):** Bağımsız test kümesi seçimi. **StratifiedGroupKFold**
  ile 5 farklı seed denenir; her biri **KL diverjansı** (train↔test dağılım benzerliği)
  + grup örtüşmesi (0 olmalı) ile puanlanır. En düşük KL'li, sıfır örtüşmeli seed
  seçilir (seed 42; tüm seedler KL=0.0006 ile çok kararlı çıktı, 163/41 bölünme).

> **Bağlantı:** Holdout, asıl modelleme yapılmadan önce ayrılır ve hiçbir tuning'de
> kullanılmaz → sızıntısız (leakage-free) nihai değerlendirme için.

---

## Aşama 02–03 — EDA ve Görselleştirme (`eda.py`, `visualize.py`)

Modelleme öncesi veri tanıma. **Bulgular sonraki algoritma seçimlerini doğrudan
yönlendirir:**
- Klorofil/NBI yaklaşık normal → klasik regresyon (Ridge/PLS) uygun.
- Flavonol sağa çarpık, dar varyans → **log-transform** kararı (`log_transform_targets`).
- Stres sınıfları dengesiz (FD=80, abiyotik=28) → **SMOTE+Tomek + class_weight** kararı.

---

## Aşama 04–05b — Öznitelik Önemi ve Seçimi (SHAP, RFE, PLS-VIP)

Üç farklı algoritma, "hangi bantlar önemli?" sorusunu **bağımsız açılardan** yanıtlar;
birbirini teyit eder (çapraz doğrulama mantığı):

| Aşama | Algoritma | Nasıl çalışır | Ne sağlar |
|---|---|---|---|
| 04 | **SHAP** | Oyun-teorisi (Shapley değerleri) ile her tahminde her bandın katkısı | Yön + büyüklük (yorumlanabilirlik) |
| 05 | **RFE** | Modeli iteratif eğitip en zayıf bandı tekrar tekrar atar | Kompakt alt küme |
| 05b | **PLS-VIP** | PLS latent bileşenlerinde değişken önem projeksiyonu | Spektroskopiye özgü önem |

> **Bağlantı:** Bu üçünün ortak işaret ettiği bölgeler (NIR R800, red-edge R700–720,
> mavi flavonoid bölgesi) daha sonra **GA (Aşama 12)** ve **konsensüs (12b/13b)**
> tarafından doğrulanır. Yani "önem" dört ayrı yöntemle (SHAP+RFE+VIP+GA) sağlamlaştırılır.

---

## Aşama 06 / 06b — Regresyon ve İki Aşamalı Tuning

**06 baseline (`m06_models/regression/`):** 6 model varsayılan hiperparametrelerle.
**06b tuned (`tuning.py`):** Her modele iki aşamalı optimizasyon.

**Modellerin rolü ve neden seçildikleri:**
| Model | Algoritma mantığı | Bu projedeki rolü |
|---|---|---|
| **PLS** | Kolineer spektrumda latent bileşen bulma — spektroskopinin altın standardı | Klorofil & NBI'da en iyi; GA ile flavonolün ana modeli |
| **Ridge** | L2 düzenlemeli lineer | Hızlı, sağlam baseline |
| **Random Forest** | Ağaç topluluğu, non-lineer, ölçek-bağımsız | Etkileşim yakalama |
| **LightGBM** | Gradient boosting | Flavonol baseline'da en iyi; hız |
| **Huber** | Aykırı-dirençli lineer | Dayanıklılık kontrolü |
| **Stacking** | Ridge+RF+LGBM → meta-Ridge | Model çeşitliliğini birleştirme |

**Tuning zinciri (algoritmalar nasıl bağlanıyor):**
```
RandomizedSearchCV (coarse, 40 iter, grup-bilinçli iç CV)
        │  en iyi bölgeyi bul
        ▼
Optuna (fine, Bayesian TPESampler + MedianPruner, 50 deneme)
        │  coarse'un etrafında daralt; pencere grid sınırına kliplenir
        ▼
fine_best_params.json  ──►  12/13/16 aşamaları bu HP'leri yeniden kullanır
```

**Sonuçlar (06b tuned, grup-bilinçli 5-kat CV, R²):**
| Hedef | En iyi model | R² | RPD | Değerlendirme |
|---|---|---|---|---|
| **Klorofil** | PLS | **0.855** | 2.62 | ✅ Mükemmel (RPD>2 = analitik) |
| **NBI** | PLS | **0.764** | 2.06 | ✅ İyi |
| **Flavonol** | LightGBM | **0.390** | 1.28 | ⚠️ Zayıf (sadece tarama) |

> **Yorum:** Klorofil/NBI'ın güçlü, geniş absorpsiyonları her modelce kolay
> yakalanır. Flavonolün UV-mavi imzası zayıf ve dar; tuning bile R²'yi ~0.39'un
> üstüne çıkaramadı. Bu, GA (Aşama 12) ihtiyacını doğurur.

---

## Aşama 07 / 07b — Stres Sınıflandırması (4 GERÇEK sınıf)

**Sınıflar:** 0=sağlıklı (40), **1=flavescence dorée/FD (80)**, 2=diğer biyotik (56;
yeşil yaprak zikadası *Empoasca vitis*, buffalo zikadası *Stictocephala bisonia*,
odun hastalıkları ve mildiyö / downy mildew *Plasmopara viticola* — külleme/powdery
mildew YOK), 3=abiyotik (28).

**Algoritma kullanımı — sınıf dengesizliği zinciri:**
```
Ham eğitim katı
   ▼ StandardScaler (yalnız train fold'a fit — leakage yok)
   ▼ SMOTE+Tomek (yalnız train fold'a — sentetik azınlık + sınır temizleme)
   ▼ class_weight="balanced"
   ▼ Model (LightGBM / RF / Stacking / SVM)
   ▼ Validation fold'da tahmin (resampling ASLA validation'a uygulanmaz)
```

**Sonuçlar (07b tuned):** En iyi klasik **LightGBM: Acc 0.804, Macro-F1 0.769**.

**FD sınıf-bazlı (LightGBM tuned) — tezin can damarı:**
| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| Sağlıklı | 0.81 | 0.88 | 0.84 |
| **FD** | **0.82** | **0.89** | **0.85** |
| Diğer biyotik | 0.81 | 0.79 | 0.80 |
| Abiyotik | 0.70 | 0.50 | 0.58 |

> **FD en iyi ayrışan sınıf** (recall %89): 80 FD vakasının 71'i doğru, yalnız 2'si
> sağlıklı sanıldı. Hastalığın spektral imzası belirgin.

> **Metodolojik not (düzeltilmiş bug):** İlk koşularda FD sınıfı (80 örnek) tamamen
> eksikti. Sebep: ground truth dosyası Fransızca (cp1252) kodlamalı; "flavescence
> dorée"deki "é" UTF-8 okumada bozuluyor (`é → �`), eşleme başarısız oluyordu.
> Encoding sağlamlaştırılarak (utf-8-sig→cp1252→latin-1) düzeltildi.

---

## Aşama 08 — Derin Öğrenme (`deep_learning/`)

7 mimari, küçük veri için **augmentation + k-fold + erken durdurma + küçük model**
stratejisiyle. Burada DL bir **üst-sınır referansı** olarak konumlanır, ana sonuç değil.

| Model | Acc | Macro-F1 | Neden bu sonuç |
|---|---|---|---|
| Autoencoder | 0.854 | 0.839 | Denoising pretrain = güçlü regularizer |
| MLP | 0.805 | 0.778 | En basit DL, sağlam |
| ResNet1D | 0.756 | 0.760 | Skip-connection |
| CNN1D | 0.732 | 0.729 | Lokal patern |
| CNN-LSTM | 0.683 | 0.644 | |
| Transformer | 0.610 | 0.586 | n=204'te attention overfit |
| RNN(LSTM) | 0.366 | 0.279 | Bantlar "zaman" değil → yanlış indüktif bias |

> **Bağlantı/yorum:** Autoencoder en yüksek (0.854) ama n=204'te DL'in
> genellenebilirliği şüpheli (Transformer/RNN'in düşüklüğü bunu kanıtlar). **Savunması
> sağlam olan klasik LightGBM (0.804) ana model olarak tercih edilir.**

---

## Aşama 09 — Ordinal Flavonol (`ordinal_flavonol.py`)

Flavonol regresyonu ~0.39'da tıkandığı için **sıra bilgisini** (düşük<orta<yüksek<PASS)
kullanan **Frank-Hall ordinal** yaklaşımı + RF/LightGBM denenir. Pratik gerekçe:
EMA ≈%3.5 eşiği (geçti/kaldı) kararı için kesin değil, bant bilgisine ihtiyaç var.

**Sonuç:** En iyi RF, 3-sınıf Acc 0.691, **QWK (Quadratic Weighted Kappa) 0.508** —
orta düzey sıralı uyum. MAE 0.314.

> **Bağlantı:** Bu, regresyon (06) ile sınıflandırma (07) arasında köprü; flavonolü
> "geçti/kaldı"ya (Aşama 07 pheur_binary) bağlayan ara temsil.

---

## Aşama 10 — Flavonol Anomali Tespiti (`anomaly_flavonol.py`)

İlaç kalite kontrolü perspektifi: "normal" = kabul edilebilir hammadde, "anomali" =
şüpheli. **Tek-sınıflı öğrenme** (one-class) — anomali örneği etiketlemeye gerek yok.

| Dedektör | Precision | Recall | F1 | Mantık |
|---|---|---|---|---|
| **One-Class SVM** | 0.588 | **0.974** | 0.733 | Kernel sınır öğrenme |
| Isolation Forest | 0.548 | 0.829 | 0.660 | Ağaç tabanlı izolasyon |
| Autoencoder | 0.818 | 0.077 | 0.141 | Reconstruction error |

> **Yorum:** One-Class SVM **recall 0.974** → şüpheli hammaddeyi neredeyse hiç
> kaçırmıyor. Düşük precision kabul edilebilir ("işaretle → laboratuvarda tekrar test").

---

## Aşama 11 — Ensemble (`ensemble.py`)

Tüm aşama sonuçlarını 3 stratejiyle birleştirir: **Direct** (hedef başına en iyi
model), **Dual** (regresyon+sınıflandırma birlikte), **Weighted** (R²-ağırlıklı
ortalama, eşik altı modeller elenir).

---

## Aşama 12 / 12b — Genetik Algoritma ile Bant Seçimi ve Konsensüs ⚠️

**Neden GA?** 520 öznitelikten flavonol için en bilgili alt küme aranıyor; brute-force
imkânsız (2⁵²⁰). **GA (DEAP)**, ikili maske üzerinde evrimsel arama yapar.

**Konfigürasyon:** Popülasyon 150, **150 nesil**, tournament seçim (k=3), uniform
çaprazlama, bit-flip mutasyon. **Fitness = grup-bilinçli 5-kat CV R²** (leakage yok).

### ⚠️ KRİTİK: Selection Bias (TEZDE MUTLAKA AÇIKLANMALI)

GA, maskeyi CV skorunu *maksimize edecek şekilde* seçer; sonra aynı CV'yi raporlar.
Bu **klasik feature-selection bias**'tır — grup-CV kullanılsa bile iyimserdir.

| Değerlendirme | Flavonol R² | Açıklama |
|---|---|---|
| GA[PLS]+PLS (12 raporu) | **0.643** | ⚠️ ŞİŞMİŞ |
| F2 biased (GA+Tuned PLS) | 0.594 | ⚠️ ŞİŞMİŞ |
| **F2n nested-CV (DÜRÜST)** | **0.283** | ✅ bias yok |
| **F3n nested-CV (DÜRÜST)** | **0.341** | ✅ bias yok |
| 06b full-feature (GA yok) | 0.390 | ✅ referans tavan |

> **ALTIN KURAL:** GA, flavonol R²'sini full-feature modellerin üstüne **dürüstçe
> ÇIKARMIYOR.** Görünen 0.30→0.64 artışı tamamen selection bias. Raporda/savunmada
> **nested-CV (F\*n) ve 06b** sayıları kullanılmalı. Dürüst flavonol R² aralığı: **~0.28–0.39**.

### GA'nın GERÇEK değeri — Yorumlanabilirlik (12b konsensüs)

GA'nın katkısı R² değil, **hangi dalga boylarının sinyal taşıdığını** göstermesidir.
Modellerin ortak (konsensüs) seçtiği bantlar (`12b`):
- **406–476 nm (mavi):** Karotenoid/flavonoid absorpsiyonu — doğrudan hammadde sinyali
- **490–549 nm (yeşil):** Antosiyanin/flavonoid yansıması
- **637–718 nm (kırmızı + red-edge):** Klorofil + bitki sağlığı

> Bu ~12 konsensüs bandı, ucuz bir **multispektral sensör** tasarımının reçetesidir
> (204 bant yerine ~12 bant). SHAP/RFE/VIP bulgularıyla örtüşür → çapraz doğrulanmış.

---

## Aşama 13 / 13b — Flavonol Kombinasyon Stratejileri ve Konsensüs

GA+PLS sonucunu aşmak için 6 strateji (`flavonol_combos.py`): residual stacking,
GA-üstü meta-stacking, boosting, ön işleme × taze-GA × PLS varyantları, multiseed GA,
regresyon→ordinal. **13b** öznitelik konsensüsünü birleştirir.

**Sonuç (en iyiler, OOF dürüst):** Stacking (GA, meta-Ridge) ΔR²_OOF=+0.027, çoğu
strateji ΔR²_OOF negatif → karmaşıklık eklemek faydasız. **multiseed bulgusu kritik:**
seed42→0.564, seed123→0.479, seed7→0.434 → seed'e göre 0.13 oynama = GA gürültüye fit
ediyor (selection bias'ın bağımsız kanıtı).

---

## Aşama 14 — Konsolide Model Özeti (`model_summary.py`)

Tüm aşamaların metriklerini tek dosyada toplar: `outputs/14_model_summary/all_models.csv`
(**10 aşama, 101 model satırı**). 16_final_combos satırları biased/nested ayrımıyla
etiketli. Tezdeki ana karşılaştırma tablosunun kaynağı.

---

## Aşama 15 — Ablation Çalışması (`ablation.py`)

"Her bileşen ne kadar katkı sağlıyor?" — bileşenleri tek tek kapatıp ölçer (LightGBM
regresyon / RF sınıflama, grup-CV, mean±std):

| Grup | Bulgu |
|---|---|
| **A. Ön işleme** | SavGol türevi kritik (B0 0.350 → A1 savgol-off 0.227). SNV etkisi hafif. |
| **B. Öznitelik** | **1. türev tek başına en güçlü** (B4=0.317) > spektrum (0.216) > indeks (0.214) |
| **C. İndeks** | ZTM ve SIPI flavonol için en bilgili indeksler |
| **D. Sınıf dengeleme** | SMOTE net katkı (RF: 0.708 → 0.751); Tomek marjinal |
| **E. Tuning** | Coarse büyük katkı (PLS: 0.259 → 0.394); fine ek katkı vermedi |

> **Bağlantı:** Ablation, Aşama 01 (ön işleme), 06b (tuning), 07 (dengeleme)
> kararlarını sayısal olarak gerekçelendirir.

---

## Aşama 16 — Final Kombinasyonlar + Nested-CV (`final_combos.py`)

5 spesifik pipeline (F1–F5) + her birinin **dürüst nested-CV karşılığı** (F1n–F4n).
**Nested-CV'nin amacı:** GA'yı dış K-katın *her train'inde tazeden* çalıştırıp
held-out'ta ölçmek → selection bias'ı yok etmek.

| Kod | Pipeline | Skor | Tip |
|---|---|---|---|
| F5 | İndeks + Tuned RF (stres) | Macro-F1 **0.844** | biased |
| F2 | GA[PLS]+Tuned PLS | R² 0.594 | biased ⚠️ |
| **F3n** | GA[LGBM] nested | R² **0.341** | ✅ dürüst |
| **F2n** | GA[PLS] nested | R² **0.283** | ✅ dürüst |
| F4 | 1.türev∩GA+PLS | R² 0.145 | biased |

`Δbias = biased − nested`: F2 = +0.311 (en büyük şişme). Bu aşama, **Aşama 12'nin
0.643'ünün neden güvenilmez olduğunu kanıtlayan kontrol mekanizmasıdır.**

---

## Genel Sonuçlar

### Hangi hedef kolay/zor?
| Hedef | En iyi R² (dürüst) | Model | Sonuç |
|---|---|---|---|
| Klorofil | **0.855** | PLS (tuned) | ✅ Kolay — güçlü, geniş absorpsiyon |
| NBI | **0.764** | PLS (tuned) | ✅ İyi |
| Flavonol | **~0.30–0.39** | LightGBM / GA-nested | ⚠️ Zor — zayıf UV-mavi imza |

### Ana hedefe ulaşıldı mı? → EVET
```
Spektrum → Hastalık tespiti (FD recall %89, genel acc %80)
        → Hastalığın flavonolü ARTIRdığı gösterildi (FD %56 geçer vs sağlıklı %30)
        → Geçti/Kaldı kararı (EMA eşiği %3.5: acc %76, KALDI recall %87)
```

### En önemli bilimsel bulgu
**FD hastalığı flavonolü DÜŞÜRMÜYOR, ARTIRIYOR** (FD ort. 3.51 vs sağlıklı 3.27;
geçme %56 vs %30). Flavonoidler savunma bileşiği olduğundan bitki strese girince
üretimini artırır. Pratik sonuç: hastalıklı yaprak paradoksal olarak daha değerli
hammadde olabilir.

### Ne çalıştı / çalışmadı?
**Çalıştı:** Klorofil/NBI PLS; stres LightGBM; FD tespiti; geçti/kaldı; anomali için
One-Class SVM; ablation/nested-CV ile dürüst metodoloji.
**Çalışmadı/sınırlı:** Flavonolün kesin sayısal tahmini (R²~0.30); GA'nın görünür R²
artışı (selection bias); RNN/Transformer (küçük veri); abiyotik sınıf (28 örnek).

### Tasarım kararları
- **Grup-bilinçli 5-kat CV** (leaf düzeyi; plot düzeyi 10 grup ile en katı test mevcut)
- **log_transform** flavonol için
- **SMOTE+Tomek yalnız train fold'a** (leakage yasağı)
- **İki aşamalı tuning** (RandomizedSearchCV → Optuna)
- **Nested-CV + ablation** ile dürüstlük güvencesi
- Tüm HP'ler `config/default.yaml`'da merkezi

---

## Pipeline Yapısı (23 aşama)

```
01_dataset                  → X (204×520), y_chl/flav/nbi/stress, groups
01b_outliers                → IQR aykırı değer raporu
01c_holdout                 → 5-seed KL ile bağımsız test seçimi (seed 42)
02_eda                      → dağılımlar, korelasyonlar
03_visualization            → RGB + indeks haritaları
04_feature_shap             → SHAP önemleri
05_feature_rfe              → RFE alt küme
05b_pls_vip                 → PLS-VIP önemleri
06_regression               → PLS, Ridge, RF, LightGBM, Huber, Stacking
07_classification           → 4 sınıf + flavonoid PASS/FAIL binary (EMA ≈%3.5, SMOTE+Tomek)
06b_regression_tuned        → RandomizedSearch + Optuna
07b_classification_tuned    → RandomizedSearch + Optuna
08_deep_learning            → MLP, CNN1D, ResNet1D, RNN, Transformer, CNN-LSTM, Autoencoder
09_ordinal_flavonol         → Frank-Hall + RF/LGBM (3 & 4 sınıf)
10_anomaly_flavonol         → IsolationForest, OneClassSVM, Autoencoder
11_ensemble                 → Direct / Dual / Weighted
12_ga_feature_selection     → GA (DEAP) bant seçimi (pop=150, ngen=150)
12b_ga_wavelength_consensus → konsensüs dalga boyu görselleri
13_flavonol_combos          → 6 kombinasyon stratejisi
13b_feature_consensus       → öznitelik konsensüsü
14_model_summary            → konsolide tablo (101 satır)
15_ablation                 → bileşen katkı analizi (A–E)
16_final_combos             → F1–F5 + nested-CV (selection bias kanıtı)
```

---

## Sunum için kısa mesaj

> **Asma yaprağının (*Vitis vinifera*) hiperspektral görüntüsünden (n=204, 204 bant)
> klorofil, flavonol, NBI içeriği ve 4 stres sınıfı tahmin edildi; 30+ model
> grup-bilinçli çapraz doğrulama ile sistematik karşılaştırıldı.**
>
> **Klorofil (R²=0.855) ve NBI (R²=0.764) PLS ile çok iyi tahmin edildi.** Stres
> sınıflandırmasında **LightGBM (Acc 0.80)** en sağlam model; özellikle **flavescence
> dorée %89 duyarlılıkla** tespit edildi.
>
> **En önemli bulgu:** FD hastalığı flavonolü artırıyor (geçme %56 vs sağlıklı %30) —
> hastalıklı yaprak daha değerli ilaç hammaddesi olabilir. **Geçti/kaldı kararı (EMA Eşiği
> ≥3.5) %76 doğrulukla** veriliyor.
>
> **Metodolojik katkı:** Genetik algoritmanın görünür R² artışının (0.64) **selection
> bias** olduğu nested-CV ile gösterildi; dürüst değer ~0.30. GA'nın asıl değeri R²
> değil, ucuz multispektral sensör için **dalga boyu seçimidir.**

---

*Rapor güncelleme: 2026-05-31 (FD-dahil tam çalıştırma)*
*Pipeline çıktıları: `outputs/01_dataset` … `outputs/16_final_combos`*
*Sonuç-odaklı detay derlemesi: `SONUC_OZETI.md`*
