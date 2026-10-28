# Loan Approval Prediction: Feature Engineering & Data Preprocessing

**AnalystLab Africa Data Science Internship, Week 2**
Author: *Your Name* | Dataset: [Loan Prediction Problem Dataset (Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

## Project Overview
A lender must decide whether each home-loan application should be approved. This project prepares the loan application data for machine learning by cleaning it, engineering affordability features, handling outliers, encoding and scaling variables, and selecting the most useful features. Every decision is documented in the notebook and in two reports.

- **Target variable:** `loan_status` (1 = approved, 0 = not approved)
- **Raw data:** 614 applications, 13 columns (7 columns contain missing values)
- **Goal:** a clean, fully numeric, machine-learning-ready dataset

## Repository Structure
```
.
├── README.md
├── requirements.txt
├── AnalystLab_Week2_Loan_Feature_Engineering.ipynb   # full workflow with outputs
├── data/
│   ├── raw/                  # original Kaggle file (copy saved by the notebook)
│   └── processed/
│       ├── cleaned_loan_data.csv        # cleaned, human-readable
│       ├── ml_ready_loan_data.csv       # encoded, scaled, feature-selected
│       └── preprocessing_summary.txt    # auto-generated results summary
├── reports/
│   ├── Business_Understanding_Report.docx
│   └── Data_Preprocessing_Report.docx
└── images/                   # figures produced by the notebook
```

## How to Run
1. Download the dataset from Kaggle and place the training CSV (the one with a `Loan_Status` column) in the project folder, or in `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook in Jupyter or Google Colab and choose **Run All**. In Colab it will prompt you to upload the CSV if it cannot find it.

## Preprocessing Pipeline
| Stage | What was done | Why |
|---|---|---|
| Inspection | Shape, dtypes, missing values, duplicates, summary stats | Understand data quality first |
| Cleaning | Renamed columns, dropped `Loan_ID`, fixed `Dependents` "3+", imputed missing values (mode / median), corrected dtypes | Valid, complete data |
| Feature engineering | `total_income`, `has_coapplicant`, `emi`, `emi_to_income`, `balance_income`, `credit_history_missing` | Reflect how lenders judge affordability |
| Outliers | Boxplots, IQR and Z-score detection; log transform + IQR capping; no rows deleted | Tame skew without losing real applicants |
| Encoding | Label Encoding (binary columns, target); One-Hot Encoding (`property_area`) | Right technique for ordered/binary vs. unordered categories |
| Scaling | `RobustScaler` on continuous features | Resistant to remaining extreme values |
| Feature selection | Correlation heatmap, Random Forest and permutation importance; dropped superseded and redundant columns | Compact, informative feature set |

## Visualizations
The notebook produces 11 labelled figures in `images/`, including missing values, target distribution, income/loan distributions, approval by credit history, scatter plot, box plots before and after treatment, scaling comparison, correlation heatmap and feature importance.

## Key Results
See `data/processed/preprocessing_summary.txt` (generated when the notebook runs) for the final shape, top features, dropped features and smoke-test scores.

## Limitations
- Small dataset (~600 rows) with imbalanced classes. Use stratified splits and report F1 / ROC-AUC.
- Imputers and scalers were fitted on the full dataset for convenience. Fit them on training data only when modelling.
- `emi` ignores interest and is a proxy.
- `gender` and `married` are sensitive attributes and need a fairness review before any real-world use.

## Next Steps
Train and compare classification models (Week 3), tune them, and evaluate fairness.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn

#AnalystLabAfrica
