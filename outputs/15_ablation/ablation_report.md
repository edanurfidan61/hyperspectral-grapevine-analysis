# Ablation Raporu (15_ablation)

Toplam deney satırı: **93**
Süre: **2726.9s**

## BASE
### classification (en iyi 3, Macro_F1)
| code   | model         | experiment     | score           |
|:-------|:--------------|:---------------|:----------------|
| B0     | lightgbm      | Baseline (tam) | 0.7444 ± 0.0365 |
| B0     | random_forest | Baseline (tam) | 0.7011 ± 0.0777 |

### regression (en iyi 3, R2)
| code   | model         | experiment     | score           |
|:-------|:--------------|:---------------|:----------------|
| B0     | random_forest | Baseline (tam) | 0.3524 ± 0.0558 |
| B0     | lightgbm      | Baseline (tam) | 0.3495 ± 0.0722 |
| B0     | pls           | Baseline (tam) | 0.2608 ± 0.1210 |

## CLS
### classification (en iyi 3, Macro_F1)
| code   | model    | experiment       | score           |
|:-------|:---------|:-----------------|:----------------|
| D1     | lightgbm | Resampling YOK   | 0.7714 ± 0.0396 |
| D2     | lightgbm | Sadece SMOTE     | 0.7575 ± 0.0359 |
| D3     | lightgbm | Class weight YOK | 0.7444 ± 0.0365 |

## FEATURE
### classification (en iyi 3, Macro_F1)
| code   | model         | experiment      | score           |
|:-------|:--------------|:----------------|:----------------|
| B2     | random_forest | Sadece indeks   | 0.7957 ± 0.0682 |
| B2     | lightgbm      | Sadece indeks   | 0.7560 ± 0.0068 |
| B1     | lightgbm      | Sadece spektrum | 0.6923 ± 0.0704 |

### regression (en iyi 3, R2)
| code   | model         | experiment      | score           |
|:-------|:--------------|:----------------|:----------------|
| B4     | random_forest | Sadece 1. türev | 0.3312 ± 0.0948 |
| B4     | lightgbm      | Sadece 1. türev | 0.3168 ± 0.1311 |
| B1     | random_forest | Sadece spektrum | 0.2909 ± 0.1404 |

## INDEX
### classification (en iyi 3, Macro_F1)
| code   | model    | experiment     | score           |
|:-------|:---------|:---------------|:----------------|
| C6     | lightgbm | SIPI çıkarıldı | 0.7653 ± 0.0459 |
| C2     | lightgbm | ARI çıkarıldı  | 0.7601 ± 0.0704 |
| C3     | lightgbm | CRI çıkarıldı  | 0.7410 ± 0.0326 |

### regression (en iyi 3, R2)
| code   | model    | experiment    | score           |
|:-------|:---------|:--------------|:----------------|
| C5     | lightgbm | ZTM çıkarıldı | 0.3674 ± 0.0656 |
| C2     | lightgbm | ARI çıkarıldı | 0.3659 ± 0.0745 |
| C3     | lightgbm | CRI çıkarıldı | 0.3626 ± 0.0596 |

## PREPROC
### classification (en iyi 3, Macro_F1)
| code   | model    | experiment        | score           |
|:-------|:---------|:------------------|:----------------|
| A1     | lightgbm | SavGol kapalı     | 0.7899 ± 0.0415 |
| A3     | lightgbm | SavGol+SNV kapalı | 0.7540 ± 0.0473 |
| A2     | lightgbm | SNV kapalı        | 0.7523 ± 0.0627 |

### regression (en iyi 3, R2)
| code   | model         | experiment    | score           |
|:-------|:--------------|:--------------|:----------------|
| A2     | random_forest | SNV kapalı    | 0.3545 ± 0.0638 |
| A2     | lightgbm      | SNV kapalı    | 0.3372 ± 0.0814 |
| A1     | random_forest | SavGol kapalı | 0.2818 ± 0.1073 |

## TUNING
### classification (en iyi 3, Macro_F1)
| code   | model    | experiment               | score           |
|:-------|:---------|:-------------------------|:----------------|
| E3     | lightgbm | Coarse + fine            | 0.7815 ± 0.0458 |
| E2     | lightgbm | Sadece coarse            | 0.7787 ± 0.0524 |
| E1     | lightgbm | HP yok (sklearn default) | 0.7444 ± 0.0365 |

### regression (en iyi 3, R2)
| code   | model    | experiment    | score           |
|:-------|:---------|:--------------|:----------------|
| E2     | pls      | Sadece coarse | 0.3935 ± 0.1780 |
| E3     | pls      | Coarse + fine | 0.3935 ± 0.1780 |
| E3     | lightgbm | Coarse + fine | 0.3902 ± 0.0473 |

