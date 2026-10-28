# Loan Approval Prediction: Feature Engineering & Data Preprocessing

**AnalystLab Africa Data Science Internship, Week 2**
Author: *Charity Nduati* | Dataset: [Loan Prediction Problem Dataset (Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

## Project Overview
A lender must decide whether each home-loan application should be approved. This project turns the raw loan application data into a machine-learning-ready dataset: cleaning it, engineering affordability features, handling outliers, encoding and scaling variables, and selecting the most useful features. Every decision is documented in the notebook and in two reports.

- **Target variable:** `loan_status` (1 = approved, 0 = not approved)
- **Raw data:** 614 applications, 13 columns, 7 columns with missing values, no duplicates
- **Result:** a fully numeric dataset of 614 rows and 11 columns (10 features + target) with no missing values

## Repository Structure
```
.
├── README.md
├── requirements.txt
├── AnalystLab_Week2_Loan_Feature_Engineering.ipynb   # full workflow, outputs included
├── data/
│   ├── raw/
│   │   └── train_u6lujuX_CVtuZ9i.csv        # original Kaggle training file
│   └── processed/
│       ├── cleaned_loan_data.csv            # cleaned, human-readable
│       ├── ml_ready_loan_data.csv           # encoded, scaled, feature-selected
│       └── preprocessing_summary.txt        # auto-generated results summary
├── reports/
│   ├── Business_Understanding_Report.docx
│   └── Data_Preprocessing_Report.docx
└── images/                                  # 11 figures produced by the notebook
```

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open the notebook in Jupyter, VS Code or Google Colab and choose **Run All**. It finds the dataset in `data/raw/` automatically. In Colab, upload `train_u6lujuX_CVtuZ9i.csv` when prompted.

The unlabeled Kaggle test file is not used, because it has no `Loan_Status` column.

## Preprocessing Pipeline
| Stage | What was done | Why |
|---|---|---|
| Inspection | Shape, dtypes, missing values, duplicates, summary stats | Understand data quality first |
| Cleaning | Renamed columns, dropped `Loan_ID`, fixed `Dependents` "3+", imputed missing values (mode / median), corrected dtypes | Valid, complete data |
| Feature engineering | `total_income`, `has_coapplicant`, `emi`, `emi_to_income`, `balance_income`, `credit_history_missing` | Reflect how lenders judge affordability |
| Outliers | Box plots, IQR and Z-score detection; log transform + IQR capping; no rows deleted | Tame skew without losing real applicants |
| Encoding | Label Encoding (binary columns, target); One-Hot Encoding (`property_area`) | Right technique for binary vs. unordered categories |
| Scaling | `RobustScaler` on continuous features | Resistant to remaining extreme values |
| Feature selection | Correlation heatmap, Random Forest and permutation importance | Compact, informative feature set |

## Key Results
- **Credit history is the dominant predictor.** In the cleaned data, 79% of applicants with a good credit history were approved versus 8% of those without. It is also the top feature in Random Forest and permutation importance.
- **Outliers:** skewness of money columns fell from as high as 11.3 to below 0.5 after log transform and capping, leaving 0 IQR outliers in the treated columns.
- **Redundancy removed:** 6 raw money columns, plus `log_balance_income` (r = 0.99 with `log_total_income`) and `log_emi` (r = 0.88 with `log_loan_amount`).
- **Low-signal features removed:** `gender`, `education`, `self_employed`, `credit_history_missing`.
- **Final features (10):** `married`, `dependents`, `loan_term_months`, `credit_history`, `has_coapplicant`, `emi_to_income`, `log_total_income`, `log_loan_amount`, `property_area_Semiurban`, `property_area_Urban`.
- **Smoke test (5-fold CV, ROC-AUC):** Logistic Regression 0.771, Random Forest 0.772, versus 0.500 for the "always approve" baseline.

## Limitations
- Small dataset (614 rows) with imbalanced classes (68.7% approved). Use stratified splits and report F1 and ROC-AUC.
- Imputers and scalers were fitted on the full dataset for convenience. Fit them on training data only when modelling.
- `emi` ignores interest and is only a proxy.
- `married` is a sensitive attribute that was retained by the selection rules. Review fairness before any real-world use.

## Next Steps
Train and compare classification models (Week 3), tune them, and evaluate fairness.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn

#AnalystLabAfrica
