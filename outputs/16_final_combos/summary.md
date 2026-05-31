# 16_final_combos — Özet

Toplam pipeline: **9**

## Sonuçlar (mean ± std)

| Kod | Görev | Hedef | Model | n_feat | Skor |
|-----|-------|-------|-------|--------|------|
| F1 | regression | flavonol | pls | 151 | 0.2406 ± 0.1128 |
| F2 | regression | flavonol | pls | 151 | 0.5939 ± 0.0553 |
| F3 | regression | flavonol | lightgbm | 254 | 0.4667 ± 0.0513 |
| F4 | regression | flavonol | pls | 45 | 0.1446 ± 0.4439 |
| F5 | classification | stress | random_forest | 112 | 0.7575 ± 0.0839 |
| F1n | regression | flavonol | pls | 205.2 | 0.2518 ± 0.1140 |
| F2n | regression | flavonol | pls | 205.2 | 0.2829 ± 0.1934 |
| F3n | regression | flavonol | lightgbm | 253.6 | 0.3413 ± 0.0757 |
| F4n | regression | flavonol | pls | 67.6 | 0.1148 ± 0.3361 |

## Yorum

- **F1 vs F2**: GA[PLS]+PLS baseline vs GA[PLS]+Tuned PLS — tuning'in
  GA-seçili dar feature kümesinde fark yaratıp yaratmadığı.
- **F3**: GA[LightGBM]+Tuned LightGBM — tuning'de en iyi sınıflandırma
  veren modelin GA ile birleşik performansı.
- **F4**: 1.türev ∩ GA[PLS] + Tuned PLS — proje_ozeti'ndeki tek-başına
  en faydalı blok (1.türev) GA ile birleşince ne olur.
- **F5**: Sadece indeks + Tuned RF — stres sınıflandırmasında indeks
  bloğunun başına yeterli mi.

## Selection bias uyarısı (F1n..F4n nested-CV)

F1..F4 satırlarındaki skor, GA maskesini **tüm veride** seçip aynı
veride CV ile değerlendirdiği için **optimistik biased**'tır (klasik
feature-selection bias). Bu yüzden her birinin **dürüst (nested-CV)**
karşılığı F1n..F4n olarak ayrıca raporlanır: her outer fold'un
train'inde GA tazeden çalışır (pop=100, ngen=80), val'da skorlanır.
`bias_vs_biased` kolonu = (biased mean) − (nested mean). Pozitif ve
büyük bir değer, biased skorun ne kadar şişirilmiş olduğunu gösterir.
**Raporda/savunmada F*n skorlarını öne çıkar.**
