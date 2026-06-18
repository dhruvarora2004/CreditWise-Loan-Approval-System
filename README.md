# CreditWise Loan Approval System

A machine learning project that predicts loan approval outcomes based on applicant financial and demographic data. Three classifiers are evaluated — Logistic Regression, Naive Bayes, and K-Nearest Neighbors — with Naive Bayes emerging as the best-balanced model for this use case.

---

## Project Structure

```
CreditWise/
│
├── src/
│   ├── preprocessing.py   # Data cleaning, imputation, encoding, feature engineering
│   ├── eda.py             # Exploratory data analysis utilities (pie charts, bar graphs)
│   ├── train.py           # Model pipelines, GridSearchCV, evaluation metrics
│   └── predict.py         # (Planned) Inference on new applicant data
│
├── notebooks/
│   └── Logic.ipynb        # End-to-end walkthrough: EDA → preprocessing → training → evaluation
│
└── README.md
```

---

## Features

**Smart Imputation**  
Missing values are handled with domain logic rather than blind statistical filling:
- `Applicant_Income` is set to the column mean for employed applicants and 0 for unemployed ones.
- `Employment_Status` is inferred from `Employer_Category` and whether income falls above or below the group median.
- Categorical columns use mode imputation; numerical columns use mean or median as appropriate.

**Feature Engineering**  
Squared income features are optionally added to capture non-linear relationships.

**Encoding**  
- `Education_Level` and `Loan_Approved` use Label Encoding (ordinal/binary).
- All other categorical columns (`Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`) use One-Hot Encoding with `drop='first'` to avoid multicollinearity.

**Model Pipelines**  
Each model is wrapped in a `Pipeline` with `StandardScaler` and tuned via `GridSearchCV` with 5-fold cross-validation, optimising for precision.

---

## Models and Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 84.7% | 0.915 | 0.557 | 0.692 |
| **Naive Bayes** | **88.2%** | **0.841** | **0.763** | **0.800** |
| KNN | 78.3% | 0.710 | 0.505 | 0.590 |

**Naive Bayes** is the recommended model. Logistic Regression achieves higher precision but its low recall means a significant number of creditworthy applicants get rejected — a costly outcome for a lending business. KNN underperforms on all metrics.

In a production loan approval system, recall is the priority metric: missing a genuinely eligible applicant is a business loss.

---

## Dataset

The dataset includes the following features:

`Applicant_ID`, `Applicant_Income`, `Coapplicant_Income`, `Employment_Status`, `Employer_Category`, `Age`, `Marital_Status`, `Dependents`, `Education_Level`, `Gender`, `Credit_Score`, `Existing_Loans`, `DTI_Ratio`, `Savings`, `Collateral_Value`, `Loan_Amount`, `Loan_Term`, `Loan_Purpose`, `Property_Area`, `Loan_Approved`



---

## Tech Stack

Python, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
