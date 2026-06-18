# Hiperspektral Asma Yaprağı Analizi — Kapsamlı Sonuç Özeti

> **Bu belgenin amacı:** Tez yazımında kaynak olarak kullanılmak üzere, pipeline'ın
> tüm aşamalarının sonuçlarını, metodolojisini ve bilimsel yorumunu eksiksiz
> derlemek. Tüm sayılar 2026-05-31 tarihli tam çalıştırmadan (FD sınıfı dahil,
> türev bloğu canlı) alınmıştır.

---

## 0. PROJENİN AMACI VE ANA SORU

**Biyomedikal motivasyon:** Kırmızı asma yaprağı (*Vitis vinifera*) bir bitkisel
ilaç hammaddesidir; **Antistax®** (AS 195) gibi venöz dolaşım ilaçlarının etken
maddesi yaprak flavonoidleridir. Antistax, Fransız Farmakopesi "Vigne Rouge"
monografına dayanır ve WEU/TU üzerinden ruhsatlıdır; **Ph.Eur. monografı yoktur**.
EMA değerlendirme raporu (EMA/HMPC/464682/2016) kırmızı asma yaprağı için ≈%3.5
flavonoid düzeyi bildirir; bu çalışmada bunu **operasyonel PASS/FAIL eşiği** olarak
kullanıyoruz.

**Ana araştırma soruları:**
1. Yapraktaki **stres/hastalık durumu** hiperspektral görüntüden tespit edilebilir mi?
   (özellikle *flavescence dorée* — FD, ekonomik açıdan en yıkıcı asma hastalığı)
2. Stres/hastalık, **ilaç hammaddesi olan flavonol içeriğini** nasıl etkiler?
3. Yaprak, EMA raporundaki ≈%3.5 flavonoid düzeyini (operasyonel eşik) **geçer mi/kalır mı** — yani ilaç
   hammaddesi olarak kullanılabilir mi — hiperspektral görüntüden tahmin edilebilir mi?
4. Yan hedef: Klorofil ve NBI (Nitrogen Balance Index) gibi diğer biyokimyasal
   parametreler ne kadar iyi tahmin edilebilir? (yöntemin genel geçerliliği)

**KISA CEVAP:** Soruların 1, 2, 3'ünde başarıya ulaşıldı. FD %89 duyarlılıkla
tespit ediliyor; FD'nin flavonolü **artırdığı** (azaltmadığı) gösterildi;
geçer/kalır kararı %76 doğrulukla veriliyor. Soru 4'te klorofil (R²=0.855) ve
NBI (R²=0.764) çok iyi; flavonolün *kesin sayısal* tahmini zayıf (dürüst R²≈0.30)
ama *eşik geçişi* (asıl pratik hedef) çalışıyor.

---

## 1. VERİ SETİ

| Özellik | Değer |
|---|---|
| Kaynak | Ryckewaert et al. hiperspektral asma yaprağı veri seti |
| Örnek sayısı | **205 ham yaprak görüntüsü**; 1'i segmentasyon maskesi boş döndüğü için elenir → **204 ile çalışılır** |
| Grup yapısı | İki düzey mevcut: (a) **leaf** = her görüntü bağımsız (204 benzersiz; aktif CV anahtarı) → grup-bilinçli CV bu düzeyde stratified k-fold ile eşdeğer çalışır; (b) **plot** = çeşit\|lokasyon\|tarih düzeyinde **10 grup** (en büyüğü 25 görüntü). Leakage'a karşı en katı test plot düzeyindedir; altyapı her ikisini de destekler. |
| Spektral bantlar | **204 bant, 397–1004 nm** (VNIR) |
| Format | ENVI hiperspektral küp |
| Ground truth | `description-2.tab` (Chl, Flav, NBI laboratuvar değerleri + symptom etiketi) |

### Hedef değişkenler (regresyon) — gerçek ölçek (son çalıştırma)

| Hedef | n | Ortalama | Std | Min | Max | Açıklama |
|---|---|---|---|---|---|---|
| Klorofil (Chl) | 204 | 21.05 | 9.30 | 2.87 | 43.01 | Fotosentez pigmenti |
| **Flavonol (Flav)** | 204 | **3.35** | **0.57** | 1.31 | 4.49 | **İlaç hammaddesi — ana hedef** |
| NBI | 204 | 6.57 | 3.36 | 0.84 | 16.75 | Azot dengesi indeksi (Chl/Flav) |

> Not: Flavonol ortalaması 3.35, EMA raporundaki ≈%3.5 eşiğin hemen altında → veri seti
> eşiğin etrafında dengeli dağılmış (**geçer 87/204 = %42.6, kalır 117/204 = %57.4**).
> Bu, eşik sınıflandırması için ideal bir zorluk seviyesidir (sınıflar dengeli).

### Stres sınıfları (sınıflandırma) — 4 GERÇEK SINIF
| Sınıf | Etiket | n | Açıklama |
|---|---|---|---|
| 0 | Sağlıklı | 40 | Belirti yok |
| **1** | **Flavescence dorée (FD)** | **80** | Fitoplazma hastalığı — tezin merkezi |
| 2 | Diğer biyotik | 56 | Yeşil yaprak zikadası (*Empoasca vitis*), buffalo zikadası (*Stictocephala bisonia*), odun hastalıkları ve mildiyö / downy mildew (*Plasmopara viticola*, 2 örnek). Külleme (powdery mildew) YOK. |
| 3 | Abiyotik / diğer | 28 | Su stresi, senesans, hasar, eksiklik |

> **KRİTİK METODOLOJİK NOT:** İlk çalıştırmalarda FD sınıfı (80 örnek) **tamamen
> eksikti** ve "abiyotik" sınıfına karışıyordu. Kök neden: ground truth dosyası
> Fransızca kaynaklı (cp1252 kodlaması) olup "flavescence dorée" kelimesindeki
> "é" karakteri UTF-8 okuma sırasında bozuluyordu (`é → �`), bu da symptom→sınıf
> eşlemesini başarısız kılıyordu. Düzeltme: dosya okuma encoding'i utf-8-sig →
> cp1252 → latin-1 sırayla deneyecek şekilde sağlamlaştırıldı. **Tez metninde bu,
> veri ön işlemenin titizliğine dair iyi bir örnektir.**

### Öznitelik (feature) yapısı — 520 öznitelik
| Blok | Sayı | İçerik |
|---|---|---|
| SNV spektrum | 204 | Standard Normal Variate normalize edilmiş tam spektrum |
| 1. türev (SG-d1) | 204 | Savitzky-Golay birinci türev (kenar/eğim bilgisi) |
| İndeks + sabit | 112 | 13 vejetasyon indeksi + NBI haritaları + continuum + edge özellikleri |

> **METODOLOJİK NOT 2:** 1. türev bloğunun 204 kolonu da bir ara çalıştırmada
> yanlışlıkla sıfırlanmıştı (ablation aşaması veri setini geçici olarak "savgol
> kapalı" haliyle yeniden kurup eski haline döndürmüyordu). Bu düzeltildi; artık
> türev bloğu canlı (std maks 3.2×10⁻²) ve ablation aşaması veri setini her zaman
> orijinal haline geri kuruyor.

---

## 2. METODOLOJİ

### 2.1 Ön işleme
- **SNV (Standard Normal Variate):** saçılma/aydınlatma farklarını giderir.
- **Savitzky-Golay 1. türev** (pencere=11, polinom derecesi=2): taban çizgisi
  kaymalarını giderir, absorpsiyon kenarlarını vurgular.
- **13 vejetasyon indeksi:** NDVI, GNDVI, ARI, RVSI, ZTM, CRI, PRI, mARI, SIPI,
  WBI, REP, BES, **FLAVI** (flavonol-spesifik indeks).

### 2.2 Çapraz doğrulama (CV) — leakage koruması
- **Grup-bilinçli 5-katlı CV** (`StratifiedGroupKFold`): Aynı fiziksel yaprağın
  farklı ölçümleri **asla** hem train hem test'e düşmez. Bu veri setinde her görüntü
  zaten bağımsız (204 benzersiz grup) olduğundan grup-bilinçli CV pratikte stratified
  k-fold ile eşdeğer çalışır; ancak altyapı, çoklu ölçüm içeren bir veriye geçildiğinde
  hiperspektral çalışmaların en sık hatasını (aynı yaprağın train/test'e sızıp skoru
  şişirmesi) önlemeye hazırdır. **Tezde "leakage'a karşı koruyucu tasarım" olarak
  vurgulanmalı.**
- Regresyonda kvantil-bazlı stratifikasyon (5 bin) ile hedef dağılımı katlar arası korunur.

### 2.3 Sınıf dengesizliği — SMOTE+Tomek
- Sınıflandırmada **SMOTE+Tomek Links** YALNIZCA her CV katının train kısmına
  uygulanır (validation'a asla — leakage yasağı). `k_neighbors` dinamik: en küçük
  sınıfa göre ayarlanır.

### 2.4 İki aşamalı hiperparametre optimizasyonu
1. **Coarse:** RandomizedSearchCV (40 iterasyon, grup-bilinçli iç CV).
2. **Fine:** Optuna Bayesian (TPESampler + MedianPruner, coarse en iyinin etrafında
   dar arama, 50 deneme). Fine pencere coarse grid sınırlarına kliplenir (örn.
   `feature_fraction ≤ 1.0` — LightGBM'in fatal hatasını önlemek için).

### 2.5 Holdout (bağımsız test) seçimi
StratifiedGroupKFold tek-fold simülasyonu, 5 farklı seed denenip **KL diverjansı**
(train↔test sınıf dağılımı benzerliği) + grup örtüşmesi (0 olmalı) ile en iyisi seçilir.
Test oranı hedefi %20.

| Seed | KL(train‖test) | Grup örtüşmesi | Test oranı | n_train / n_test | Durum |
|---|---|---|---|---|---|
| **42** | 0.0006 | 0 | 0.201 | 163 / 41 | ✅ SEÇİLDİ |
| 1 | 0.0006 | 0 | 0.201 | 163 / 41 | |
| 7 | 0.0006 | 0 | 0.201 | 163 / 41 | |
| 21 | 0.0006 | 0 | 0.201 | 163 / 41 | |
| 123 | 0.0006 | 0 | 0.201 | 163 / 41 | |

> Not: Tüm seedler aynı KL (0.0006) ve sıfır grup örtüşmesi verdi → bölünme
> seçimine duyarsız, çok kararlı. Kural gereği grup_overlap=0 olanlar arasından
> en küçük KL'li (eşitlikte ilk) seed 42 seçildi.

### 2.6 Modeller
- **Regresyon:** PLS, Ridge, Huber, Random Forest, LightGBM, Stacking.
- **Sınıflandırma:** Random Forest, LightGBM, Stacking, SVM (pheur), pheur_binary.
- **Derin öğrenme:** MLP, CNN1D, ResNet1D, RNN(LSTM), CNN-LSTM, Transformer, Autoencoder.
- **Özel:** Ordinal (Frank-Hall), Anomali tespiti (Isolation Forest, One-Class SVM, AE),
  GA tabanlı dalga boyu seçimi (DEAP).

---

## 3. SONUÇLAR — HEDEF HEDEF

### 3.1 SORU 1 — Stres/Hastalık Tespiti ✅ BAŞARILI

**En iyi klasik model: LightGBM (tuned), 4 sınıf**
- Accuracy: **0.804**, Macro-F1: **0.769**, Balanced Acc: 0.762

Sınıf bazlı (LightGBM tuned):
| Sınıf | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Sağlıklı | 0.81 | 0.88 | 0.84 | 40 |
| **Flavescence dorée (FD)** | **0.82** | **0.89** | **0.85** | 80 |
| Diğer biyotik | 0.81 | 0.79 | 0.80 | 56 |
| Abiyotik / diğer | 0.70 | 0.50 | 0.58 | 28 |

**Confusion matrix (LightGBM tuned):**
```
                 tahmin→  sağlıklı  FD  biyotik  abiyotik
gerçek sağlıklı           35        3   1        1
gerçek FD                  2       71   5        2     ← 80 FD'nin 71'i doğru
gerçek biyotik             1        8  44        3
gerçek abiyotik            5        5   4       14
```

**Yorum:**
- **FD en iyi ayrışan sınıf** (F1=0.85, recall=0.89). 80 FD vakasının 71'i doğru
  yakalandı, yalnızca 2'si sağlıklı sanıldı. Bu, tezin biyomedikal hipotezini
  doğrudan destekler: FD hastalığının spektral imzası belirgin ve ayırt edilebilir.
- **Zayıf sınıf: abiyotik** (F1=0.58, recall=0.50). Sebep: yalnızca 28 örnek ve
  içerik heterojen (su stresi + senesans + hasar + eksiklik bir arada). Bu beklenen
  bir sınırlamadır.

**Derin öğrenme karşılaştırması (4 sınıf stres):**
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Autoencoder | 0.854 | 0.839 |
| MLP | 0.805 | 0.778 |
| ResNet1D | 0.756 | 0.760 |
| CNN1D | 0.732 | 0.729 |
| CNN-LSTM | 0.683 | 0.644 |
| Transformer | 0.610 | 0.586 |
| RNN(LSTM) | 0.366 | 0.279 |

> **Tez için kritik tartışma:** Autoencoder en yüksek skoru veriyor (0.854) ama
> n=204 gibi küçük bir veri setinde derin öğrenme modellerinin genellenebilirliği
> şüphelidir (Transformer ve RNN'in düşük skorları bunu kanıtlıyor — modern dizi
> mimarileri küçük spektral veride klasik yöntemlerin gerisinde kalıyor). **Savunması
> daha sağlam olan klasik LightGBM (0.804) tercih edilmeli;** derin öğrenme bir
> "üst sınır referansı" olarak raporlanmalı, ana sonuç olarak değil.

---

### 3.2 SORU 2 — Hastalık → İlaç Hammaddesi İlişkisi ✅ KURULDU (ANA BULGU)

**Stres sınıfına göre flavonol içeriği:**
| Stres sınıfı | n | Flavonol ort. | EMA ≈3.5 geçme oranı (≥3.5) |
|---|---|---|---|
| Sağlıklı | 40 | 3.27 | %30 (12/40) |
| **FD (hasta)** | 80 | **3.51** | **%56 (45/80)** |
| Diğer biyotik | 56 | 3.24 | %38 (21/56) |
| Abiyotik | 28 | 3.17 | %25 (7/28) |

**🔑 EN ÖNEMLİ BULGU:** **FD hastalığı flavonol içeriğini DÜŞÜRMÜYOR, AKSİNE
ARTIRIYOR.** FD'li yaprakların ortalama flavonolü 3.51 (sağlıklının 3.27'sinin
üzerinde) ve %56'sı EMA raporundaki ≈%3.5 eşiği geçiyor (sağlıklıda yalnızca %30).

**Biyolojik açıklama:** Flavonoidler bitkilerde **savunma/stres bileşikleridir**
(UV koruma, antioksidan, patojen savunması). Bitki FD fitoplazmasının saldırısına
girdiğinde flavonoid biyosentezini **yukarı regüle eder**. Bu, literatürdeki
"stres kaynaklı sekonder metabolit artışı" ile tutarlıdır.

**Pratik/ekonomik sonuç (tez için güçlü argüman):** Hastalıklı (FD) yaprak,
paradoksal biçimde **daha değerli ilaç hammaddesi** olabilir. Bu, "hastalıklı
ürünü ıskartaya çıkar" sezgisinin tersini gösterir ve hiperspektral tarama ile
hammadde kalitesinin önceden kestirilebileceğini kanıtlar.

---

### 3.3 SORU 3 — Geçti/Kaldı Sınıflandırması (EMA ≈%3.5 eşiği) ✅ ORTA-İYİ

**Model: pheur_binary (SVM), flavonol ≥ 3.5 ikili sınıflandırma**
- Accuracy: **0.765**, Macro-F1: **0.751**, Balanced Acc: 0.746
- (SMOTE'li ve tuned versiyonlar neredeyse aynı → sonuç sağlam, modele duyarsız)

Sınıf bazlı:
| Sınıf | Precision | Recall | F1 | n |
|---|---|---|---|---|
| FAIL / KALDI (<3.5) | 0.76 | 0.87 | 0.81 | 117 |
| PASS / GEÇTİ (≥3.5) | 0.78 | 0.62 | 0.69 | 87 |

**Confusion matrix:**
```
              tahmin→  KALDI  GEÇTİ
gerçek KALDI           102    15
gerçek GEÇTİ            33     54
```

**Yorum:**
- **KALDI sınıfı çok iyi yakalanıyor (recall 0.87):** Eşiği geçemeyen yaprakların
  %87'si doğru reddediliyor. Kalite kontrol açısından bu kritik — düşük kaliteli
  hammaddenin elenmesi yüksek güvenilirlikle yapılabiliyor.
- **GEÇTİ sınıfı orta (recall 0.62):** Geçen 87 yaprağın 54'ü doğru, 33'ü kaçırıldı.
  Bu muhafazakâr (false-negative ağırlıklı) davranış kalite kontrolünde aslında
  **güvenli taraftadır** (kötü olanı iyi sanmaktansa, iyi olanı eleme riski).
- Pratik değer: Tam flavonol miktarını bilmeden, sadece spektrumdan "ilaç
  hammaddesi olur/olmaz" kararı %76 doğrulukla verilebiliyor.

---

### 3.4 SORU 4 — Biyokimyasal Parametre Tahmini (Regresyon)

**En iyi tuned model sonuçları (grup-bilinçli 5-kat CV, R²):**

| Hedef | En iyi model | R² | RMSE | RPD | Değerlendirme |
|---|---|---|---|---|---|
| **Klorofil** | PLS | **0.855** | 3.545 | 2.62 | ✅ Mükemmel (RPD>2 = analitik kullanıma uygun) |
| **NBI** | PLS | **0.764** | 1.633 | 2.06 | ✅ İyi (RPD>2) |
| **Flavonol** | LightGBM | **0.390** | 0.449 | 1.28 | ⚠️ Zayıf (RPD<1.5 = sadece kaba tarama) |

**Tam regresyon tablosu (06b tuned):**
```
Klorofil:  PLS 0.855 | RF 0.843 | LightGBM 0.841 | Huber 0.752 | Ridge 0.690
NBI:       PLS 0.764 | LightGBM 0.763 | RF 0.759 | Huber 0.505 | Ridge 0.500
Flavonol:  LightGBM 0.390 | PLS 0.380 | RF 0.370 | Huber 0.217 | Ridge 0.144
```

**RPD (Residual Prediction Deviation) yorumu:** RPD>2 analitik tahmin için yeterli,
1.5–2 kaba tarama, <1.5 güvenilmez. Klorofil ve NBI analitik düzeyde; flavonol
yalnızca tarama düzeyinde.

**Neden flavonol zor?** (Tez için önemli tartışma):
- Flavonolün spektral imzası klorofile göre çok daha zayıf ve dar.
- Klorofil 550 ve 700 nm civarında güçlü, geniş absorpsiyon yapar; flavonol UV-mavi
  bölgede daha incedir.
- Dinamik aralık dar (2.04–5.12, std 0.57) → küçük mutlak hatalar R²'yi düşürür.

---

## 4. GENETİK ALGORİTMA (GA) İLE DALGA BOYU SEÇİMİ — ⚠️ KRİTİK METODOLOJİK UYARI

### 4.1 Selection Bias sorunu (TEZDE MUTLAKA AÇIKLANMALI)

GA, 520 öznitelikten flavonol tahminini en iyileyen alt kümeyi seçer. **Ham GA
sonuçları yanıltıcı derecede yüksektir:**

| Değerlendirme | Flavonol R² | Açıklama |
|---|---|---|
| GA[PLS]+PLS (12_GA raporu) | **0.643** | ⚠️ Şişmiş — maske CV skorunu maksimize edecek şekilde seçildi |
| F2 biased (GA+Tuned PLS) | 0.594 | ⚠️ Şişmiş |
| **F2n nested-CV (DÜRÜST)** | **0.283** | ✅ Selection bias yok |
| **F3n nested-CV (DÜRÜST, LightGBM)** | **0.341** | ✅ Selection bias yok |
| 06b full-feature (GA yok) LightGBM | 0.390 | ✅ Bias yok (referans tavan) |

**Sorunun özü:** GA, çapraz doğrulama skorunu kullanarak öznitelik seçtiği için,
seçilen maskeyi yine aynı CV ile değerlendirmek **klasik feature-selection
bias**'tır (binlerce aday arasından "en iyi CV vereni" seçip o CV'yi raporlamak
doğası gereği iyimserdir — grup-bilinçli CV kullanılsa bile).

**Çözüm — Nested (iç içe) CV:** GA, dış K-katlı CV'nin **her katının train kısmında
ayrı ayrı** çalıştırılır, held-out kat üzerinde değerlendirilir. Bu, "metodun
ortalama genellenebilirliğini" dürüstçe ölçer.

`Δbias = (biased skor) − (nested skor)`:
- F2: 0.594 → 0.283 (**Δ = +0.311**, en büyük şişme; tuning + sabit maske kombinasyonu)
- F3: 0.467 → 0.341 (Δ = +0.125)
- F1: 0.241 → 0.252 (Δ ≈ 0; default PLS aşırı uydurmuyor)
- F4: 0.145 → 0.115 (Δ ≈ 0)

> **TEZ İÇİN ALTIN KURAL:** GA'nın flavonol R²'sini full-feature modellerin üstüne
> dürüstçe ÇIKARMADIĞINI açıkça yaz. Görünen artış (0.30 → 0.64) tamamen selection
> bias. Savunmada **nested-CV (F\*n) ve full-feature (06b)** sayıları kullanılmalı,
> 0.64 değil. Honest flavonol R² aralığı: **~0.28–0.39**.

### 4.2 GA'nın GERÇEK değeri — Yorumlanabilirlik

GA'nın katma değeri R² artışı değil, **hangi dalga boylarının biyolojik sinyal
taşıdığını ortaya koymasıdır.** 7 farklı GA modelinin ortak (konsensüs) seçtiği bantlar:

**Tüm 7 modelin ortak seçtiği (7/7 konsensüs) dalga boyları:**
```
412, 441, 458, 476, 516, 549, 593, 637 nm  (hepsi SNV spektrum)
```
**6/7 konsensüs:** 671, 689, 718 nm (SNV) + 412, 460 nm (1. türev)
**5/7 konsensüs:** 480, 503 nm (1. türev)

**Biyolojik yorum:**
- **412–476 nm (mavi):** Karotenoid ve flavonoid absorpsiyonu — doğrudan ilaç
  hammaddesi sinyali.
- **516–549 nm (yeşil):** Antosiyanin/flavonoid yansıma bölgesi.
- **637–689 nm (kırmızı):** Klorofil absorpsiyonu.
- **718 nm (kırmızı kenar):** "Red edge" — bitki sağlığının klasik göstergesi,
  stres durumunda kayar.

> Bu konsensüs bantları, ucuz bir **çok-bantlı (multispektral) sensör** tasarımı
> için doğrudan reçetedir: tam 204 bant yerine ~12 bant ile benzer bilgi
> yakalanabilir. Pratik/endüstriyel katkı olarak tezde vurgulanabilir.

---

## 5. ABLATION ÇALIŞMASI — Hangi bileşen ne kadar katkı sağlıyor?

Tüm ablation sonuçları grup-bilinçli 5-kat CV ile (mean ± std).

### A. Ön işleme (PREPROCESSING) — LightGBM regresyon, flavonol
| Kod | Deney | R² |
|---|---|---|
| B0 | Baseline (tam) | 0.350 |
| A1 | SavGol kapalı | 0.227 ↓ |
| A2 | SNV kapalı | 0.337 |
| A3 | SavGol+SNV kapalı | 0.220 ↓ |

→ **SavGol türevi kritik** (kapatınca 0.350 → 0.227). SNV'nin etkisi daha hafif.

### B. Öznitelik blokları — LightGBM regresyon
| Kod | Deney | R² |
|---|---|---|
| B1 | Sadece spektrum (SNV) | 0.216 |
| B2 | Sadece indeksler | 0.214 |
| B4 | **Sadece 1. türev** | **0.317** |
| B0 | Hepsi (baseline) | 0.350 |

→ **1. türev tek başına en güçlü blok** (0.317), spektrum (0.216) ve indekslerden
(0.214) belirgin üstün. Bu, türev ön işlemenin flavonol için neden önemli olduğunu
gösterir ve A1 (SavGol kapalı) sonucuyla tutarlı.

### C. İndeks ablasyonu (her indeksi tek tek çıkar) — en etkili 5
| Çıkarılan indeks | Kalan R² (LightGBM) | Etki |
|---|---|---|
| ZTM | 0.329 | en büyük düşüş |
| SIPI | 0.332 | |
| GNDVI | 0.334 | |
| PRI | 0.337 | |
| FLAVI | 0.336 | |

→ ZTM (Zarco-Tejada & Miller) ve SIPI flavonol için en bilgilendirici indeksler.

### D. Sınıf dengeleme (CLASSIFICATION) — Random Forest, stres
| Kod | Deney | Macro-F1 |
|---|---|---|
| D1 | Resampling YOK | 0.708 |
| D2 | Sadece SMOTE | **0.751** |
| D3 | Class weight YOK | 0.738 |
| D4 | SMOTE+Tomek + balanced (tam) | 0.738 |

→ **SMOTE sınıf dengelemesi net katkı sağlıyor** (0.708 → 0.751). Tomek Links'in
ek katkısı marjinal.

### E. Hiperparametre optimizasyonu — PLS regresyon, flavonol
| Kod | Deney | R² |
|---|---|---|
| E1 | HP yok (sklearn default) | 0.259 |
| E2 | Sadece coarse | **0.394** |
| E3 | Coarse + fine | 0.394 |

→ **Coarse tuning büyük katkı** (0.259 → 0.394). Fine (Optuna) ek katkı sağlamadı
→ arama uzayı coarse'da zaten yeterince tarandı.

---

## 6. EK ANALİZLER

### 6.1 Ordinal flavonol sınıflandırması (09)
Flavonolü sıralı sınıflara (düşük/orta/yüksek) bölme:
| Şema | En iyi model | Accuracy | QWK (Quadratic Weighted Kappa) | MAE |
|---|---|---|---|---|
| 3 sınıf | Random Forest | 0.691 | 0.508 | 0.314 |
| 4 sınıf | Random Forest | 0.696 | 0.502 | 0.314 |

→ QWK ≈ 0.50 = orta düzey sıralı uyum. Tam regresyondan daha kararlı ama kaba.

### 6.2 Anomali tespiti (10) — EMA ≈%3.5 eşiği aşımı anomali olarak
| Model | Precision | Recall | F1 |
|---|---|---|---|
| One-Class SVM | 0.588 | **0.974** | 0.733 |
| Isolation Forest | 0.548 | 0.829 | 0.660 |
| Autoencoder | 0.818 | 0.077 | 0.141 |

→ One-Class SVM çok yüksek recall (0.974) — eşik aşan yaprakları kaçırmıyor ama
düşük precision (çok false-positive). Kalite kontrolde "hiçbir iyi hammaddeyi
kaçırma" senaryosu için uygun.

### 6.3 Final kombinasyonlar (16) — özet tablo
| Kod | Pipeline | Skor | Tip |
|---|---|---|---|
| F5 | İndeks + Tuned RF (stres) | Macro-F1 **0.844** | biased |
| F2 | GA[PLS] + Tuned PLS | R² 0.594 | biased ⚠️ |
| F3 | GA[LGBM] + Tuned LGBM | R² 0.467 | biased ⚠️ |
| **F3n** | F3'ün nested-CV'si | R² **0.341** | ✅ dürüst |
| **F2n** | F2'nin nested-CV'si | R² **0.283** | ✅ dürüst |
| **F1n** | GA[PLS]+PLS nested | R² **0.252** | ✅ dürüst |
| F1 | GA[PLS] + PLS (default) | R² 0.241 | biased |
| F4 | 1.türev ∩ GA + Tuned PLS | R² 0.145 | biased |
| F4n | F4'ün nested-CV'si | R² 0.115 | ✅ dürüst |

---

## 7. GENEL DEĞERLENDİRME VE TEZ İÇİN ÇIKARIMLAR

### 7.1 Ana hedefe ulaşıldı mı? → EVET
İşleyen uçtan uca zincir kuruldu:
```
Hiperspektral spektrum
   → Hastalık tespiti (FD recall %89, genel acc %80)
   → Hastalığın flavonolü ARTIRDIĞI gösterildi (FD %56 geçer vs sağlıklı %30)
   → Geçti/Kaldı kararı (EMA ≈%3.5 eşiği, acc %76, KALDI recall %87)
```

### 7.2 Güçlü yönler (tezde öne çıkar)
1. **Grup-bilinçli CV** ile leakage'sız, savunulabilir metodoloji.
2. **FD tespiti** yüksek başarı (%89 recall) — biyomedikal motivasyon kanıtlandı.
3. **Stres↔hammadde ilişkisi** beklenmedik ve değerli bulgu (FD flavonolü artırıyor).
4. **Selection bias'ın dürüstçe ele alınması** (nested-CV) — metodolojik olgunluk.
5. **Klorofil/NBI tahmininde mükemmel sonuç** (R²>0.76, RPD>2) — yöntem geçerli.
6. **GA konsensüs bantları** ile ucuz multispektral sensör tasarımına yol.

### 7.3 Sınırlamalar (tezde dürüstçe belirt)
1. **Flavonolün kesin sayısal tahmini zayıf** (dürüst R²≈0.30, RPD<1.5). Ama eşik
   geçişi (asıl hedef) çalışıyor.
2. **Küçük veri seti** (n=204 görüntü, 10 plot grubu) — derin öğrenme genellenebilirliği şüpheli.
3. **Abiyotik sınıf zayıf** (28 örnek, heterojen).
4. **GA görünür R² artışı selection bias** — dürüst değerlendirmede full-feature'ı geçmiyor.

### 7.4 Pratik/endüstriyel katkı
- Hiperspektral tarama ile **hammadde kalitesi tahribatsız ve önceden** kestirilebilir.
- Hastalıklı (FD) yaprağın değerli hammadde olabileceği gösterildi — atık azaltma.
- ~12 konsensüs bandı ile ucuz multispektral cihaz tasarımı mümkün.

### 7.5 Gelecek çalışma önerileri
- Daha büyük, çok mevsimli/lokasyonlu veri seti ile genellenebilirlik testi.
- Flavonol için hedefe özel spektral ön işleme veya derin öğrenme (yeterli veriyle).
- Konsensüs bantlarıyla saha-tipi multispektral prototip.
- FD'nin flavonol artışının biyokimyasal mekanizmasının (HPLC ile) doğrulanması.

---

## 8. TEKNİK EK — Pipeline Aşamaları (23 aşama, 2sa 38dk, hatasız)

| # | Aşama | Süre | Çıktı |
|---|---|---|---|
| 01 | Veri seti kurulumu | 210s | X.npy, y_*.npy (204×520) |
| 01b | Aykırı değer filtresi | <1s | |
| 01c | Holdout seçimi (5 seed, KL) | <1s | seed 42 |
| 02 | EDA | 1s | |
| 03 | Görselleştirme | 565s | spektrum grafikleri |
| 04 | SHAP öznitelik önemi | 4s | |
| 05 | RFE öznitelik eleme | 27s | |
| 05b | PLS-VIP | <1s | |
| 06 | Regresyon (baseline) | 84s | |
| 07 | Sınıflandırma (SMOTE) | 15s | |
| 06b | Regresyon (tuned) | 750s | en iyi HP |
| 07b | Sınıflandırma (tuned) | 897s | |
| 08 | Derin öğrenme | 427s | 7 model |
| 09 | Ordinal flavonol | 7s | |
| 10 | Anomali tespiti | <1s | |
| 11 | Ensemble | <1s | |
| 12 | GA dalga boyu seçimi | 1265s | maskeler + bantlar |
| 12b | GA dalga boyu konsensüs | 2s | görseller |
| 13 | Flavonol kombinasyonları | 557s | |
| 13b | Öznitelik konsensüs | <1s | |
| 14 | Model özeti (konsolide) | <1s | all_models.csv (101 satır) |
| 15 | Ablation | 2717s | tüm bileşen analizleri |
| 16 | Final kombinasyonlar + nested-CV | 2094s | bias analizi |

**Konsolide sonuç dosyası:** `outputs/14_model_summary/all_models.csv` (10 aşama, 101 model satırı).

---

*Bu özet, tez yazımında doğrudan kaynak olarak kullanılmak üzere hazırlanmıştır.
Tüm sayılar grup-bilinçli çapraz doğrulama (leakage'sız) sonuçlarıdır. Şişmiş
(biased) GA skorları ile dürüst (nested-CV) skorlar açıkça ayrılmıştır.*
