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
│   └── census-bureau.columns      
├── figures/
│   ├── fig1_label_distribution.png
│   ├── fig2_missing_values.png
│   ├── fig3_leakage_assessment.png
│   ├── fig4_high_income_by_variable.png
│   ├── figA_roc_curves.png
│   ├── figB_auc_comparison.png
│   ├── figC_confusion_matrix.png
│   ├── figD_feature_importance.png
│   └── figE_segments_pca.png
├── notebook/
│   └── Income_Classification_and_Segmentation.ipynb
└── report/
    └── Census_Report.pdf
```
---

## Data

**Source:** U.S. Census Bureau, Current Population Survey 1994–1995  
**Size:** 199,523 records × 40 variables + sampling weight + income label  
**Label:** Binary — above or below $50k annual income  

Place `census-bureau.data` and `census-bureau.columns` in the `data/` folder before running.

---

## Setup

```bash
pip install -r requirements.txt
```

All code runs in a single Jupyter notebook. Tested on Python 3.10+ and Google Colab.

> **Note for Colab users:** Most packages are pre-installed. You may only need `pip install xgboost`.

---

## Notebook Structure

**1. Data Loading & EDA**
- Label distribution (raw class imbalance: 6.21% high income)
- Missing value analysis — four migration columns with ~50% missing values
- Leakage assessment: `tax_filer_stat` identified and removed
- Variable separation analysis: asset ownership and education vs. demographic signals

**2. Preprocessing**
- Drop columns: migration (50% missing), redundant detailed codes, leakage variable
- Feature engineering: `weeks_worked` → 3 categories; binary flags for zero-inflated asset variables; education consolidated into 4 tiers
- Imputation, label encoding, and sanity checks
- `sample_weight` preserved separately for model training

**3. Income Classification**
- 80/20 stratified train/test split
- Baselines: Logistic Regression, Random Forest
- XGBoost with RandomizedSearchCV (30 combinations, 3-fold CV)
- Threshold optimization via precision-recall curve → optimal threshold 0.807
- Evaluation: AUC-ROC, confusion matrix, feature importance

**4. Customer Segmentation**
- Population: working-age adults 18–65 in the labor force (93,124 records)
- Features aligned with classifier importance: age, weeks_worked, education, marital status, occupation type (sex and race excluded — bias risk)
- K selection: silhouette score K=2 to K=7 → K=4 rejected (144-person micro-cluster), K=3 selected
- Cluster profiling and PCA visualization

**5. Visualization**
- All figures saved to `figures/` directory

---

## Key Results

| Model | AUC-ROC |
|---|---|
| Logistic Regression | 0.863 |
| Random Forest | 0.906 |
| XGBoost (threshold = 0.807) | 0.927 |

**At optimized threshold:** 61% precision, 6.4x lift over random baseline

| Persona | Size | High-income rate | Priority |
|---|---|---|---|
| Established Professional | 38,410 (41%) | 18.8% | Highest |
| Ambitious Full-Timer | 43,259 (47%) | 8.6% | Medium |
| Emerging Young Worker | 11,455 (12%) | 1.2% | Low-cost only |

---

## Key Design Decisions

**Why AUC-ROC over accuracy:** With 9.47% positive labels, a model predicting "low income" for everyone achieves 93.79% accuracy — completely useless for marketing.

**Why `tax_filer_stat` was dropped:** "Nonfiler" had 0% high-income rate — the column encodes income as a behavioral consequence, not an independent predictor.

**Why `sample_weight` matters:** Census surveys use stratified sampling. Without passing these weights to the model, predictions optimize for sample proportions rather than the true U.S. population.

**Why K=3 over K=4:** K=4 had a marginally higher silhouette score (0.3144 vs 0.3113) but produced a cluster of only 144 people — an artifact of `dividends_from_stocks` being 96% zero.

**Why sex and race were excluded from segmentation:** Bias risk. The income classifier still captures their predictive signal; the segmentation model uses behavioral and career variables only.

---

## References

Salminen, J. et al. (2023). How can algorithms help in segmenting users and customers? A systematic review and research agenda for algorithmic customer segmentation. Journal of Marketing Analytics. 

Herhausen, D., Bernritter, S., Ngai, E.W.T., Kumar, A., & Delen, D. (2024). Machine learning in marketing: Recent progress and future research directions. Journal of Business Research, 170, Article 114254
---

*Data source: U.S. Census Bureau, Current Population Survey, 1994–1995. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/117/census+income+kdd*
