# Census Income Classification & Customer Segmentation
**JPMorgan Chase & Co. — Data Science Take-Home Project**  
Zhilin Chen · March 2026

---

## Project Overview

This project builds a two-model targeting framework on the U.S. Census Bureau's Current Population Survey (1994–1995) for a retail business client:

1. **Income Classifier (XGBoost)** — scores every prospect by their likelihood of earning above $50k annually
2. **Customer Segmentation (K-Means)** — groups working-age adults into three actionable marketing personas

Used together, the classifier answers *whether* to invest in a prospect; the segment answers *how* to approach them.

---

## Repository Structure

```
census-income-targeting/
├── README.md
├── requirements.txt
├── data/
│   └── census-bureau.columns           # column names (raw data not included)
├── figures/
│   ├── fig1_label_distribution.png
│   ├── fig2_missing_values.png
│   ├── fig3_leakage_assessment.png
│   ├── fig4_behavioral_vs_demographic.png
│   ├── figA_roc_pr_curves.png          # ROC + Precision-Recall curves
│   ├── figB_model_comparison.png
│   ├── figC_confusion_matrix.png
│   ├── figD_feature_importance.png
│   ├── figE_shap_summary.png           # SHAP summary plot
│   └── figE2_shap_bar.png              # SHAP bar chart
├── notebook/
│   └── Income_Classification_and_Segmentation.ipynb
└── report/
    └── Census_Report.pdf
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tested on Python 3.10+ and Google Colab.

> **Note for Colab users:** Most packages are pre-installed. You may only need `pip install xgboost shap`.

---

## Notebook Structure

**Part I — Data Loading & EDA**
- Label distribution: 6.21% high income — severe class imbalance
- Missing value analysis: four migration columns (~50% missing) dropped
- Leakage assessment: `tax_filer_stat` identified and removed
- Variable separation: asset ownership and education vs. demographic signals (Fig 4)

**Part II — Preprocessing**
- Drop columns: migration (50% missing), redundant detailed codes, leakage variable
- Feature engineering:
  - `weeks_worked_in_year` → 3 categories (none / partial / full)
  - Binary flags for zero-inflated variables: `has_capital_gains`, `has_dividends_from_stocks`, etc.
  - Education consolidated from 16 categories into 4 tiers
- Census `sample_weight` preserved separately and passed to all models during training
- Label Encoding for all categorical columns (appropriate for tree-based models)

**Part III — Income Classification**
- 80/20 stratified train/test split (preserves 9.47% positive rate)
- Baselines: Logistic Regression (`class_weight='balanced'`), Random Forest (`balanced_subsample`)
- XGBoost tuned via RandomizedSearchCV: 30 combinations, 3-fold StratifiedKFold, scored on AUC-ROC
- Three simultaneous imbalance strategies: `sample_weight` + `balanced_subsample` + `scale_pos_weight=9.56`
- Threshold optimization via precision-recall curve → optimal threshold 0.807 (max F1)
- SHAP analysis for model interpretability (Figs E, E2)

**Part IV — Customer Segmentation**
- Population: working-age adults in the labor force (93,124 records)
- Features aligned with classifier importance: age, weeks_worked, education, marital status, occupation type
- Sex and race excluded — bias risk in marketing targeting
- OneHotEncoding applied to categorical features before K-Means (Euclidean distance approximation)
- K selection: silhouette score K=2 to K=7 → K=4 rejected (144-person micro-cluster artifact), K=3 selected
- Cluster profiling and PCA visualization

---

## Key Results

| Model | AUC-ROC | AUC-PR | Precision (>$50k) | Recall (>$50k) | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.863 | 0.503 | 27% | 78% | 0.40 |
| Random Forest | 0.906 | 0.588 | 33% | 80% | 0.47 |
| XGBoost (default thr=0.5) | 0.927 | 0.674 | 37% | 83% | 0.51 |
| XGBoost (optimized thr=0.807) | 0.927 | 0.674 | 61% | 61% | 0.61 |

**At optimized threshold:** 6.4x lift over random baseline — per 10,000 prospects, 61% of flagged contacts are genuinely high-income vs. 9.47% at random.

| Persona | Size | High-income rate | Marketing priority |
|---|---|---|---|
| Established Professional | 38,410 (41%) | 18.8% | Highest — premium campaigns |
| Ambitious Full-Timer | 43,259 (47%) | 8.6% | Medium — aspirational messaging |
| Emerging Young Worker | 11,455 (12%) | 1.2% | Low-cost acquisition only |

---
## Key Design Decisions

**- AUC-PR as primary metric:** With 9.47% positive labels, AUC-ROC is misleadingly optimistic. AUC-PR directly measures minority class performance — XGBoost achieves 0.674 vs. a random baseline of 0.095. Threshold 0.807 selected by maximizing F1 on the precision-recall curve.

**- Leakage removal:** `tax_filer_stat` dropped — "Nonfiler" had 0% high-income rate, encoding income as a consequence rather than a predictor.

**- Census sample weights:** Passed to all models to reflect true U.S. population distribution, not sample proportions.

**- K=3 over K=4:** K=4 silhouette (0.3144) was marginally higher but produced a 144-person micro-cluster — artifact of `dividends_from_stocks` being 96% zero. K=3 gives three balanced, interpretable personas.

**- OneHotEncoding for K-Means:** K-Means uses Euclidean distance, which is not suited for categoricals. OneHotEncoding converts categories to binary dimensions as a standard approximation. K-Modes and K-Prototypes were considered but rejected due to sklearn compatibility and added hyperparameter complexity.

**- Sex and race excluded from segmentation:** Bias risk. Predictive signal is already captured by the income score — segmentation uses behavioral and career variables only.
---

## References

Salminen, J., Mustak, M., Corporan, J., & Jansen, B.J. (2023). How can algorithms help in segmenting users and customers? A systematic review and research agenda for algorithmic customer segmentation. *Journal of Marketing Analytics, 11*, 438–467.

Herhausen, D., Bernritter, S., Ngai, E.W.T., Kumar, A., & Delen, D. (2024). Machine learning in marketing: Recent progress and future research directions. *Journal of Business Research, 170*, Article 114254.
