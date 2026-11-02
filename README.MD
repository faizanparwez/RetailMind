# RetailMind AI

## Product Vision
RetailMind AI is an incremental machine learning project focused on building intelligent solutions for retail businesses.

## Day 1 — Customer Churn Prediction

### Objective
Build an end-to-end machine learning pipeline to predict whether a telecom customer is likely to leave the service.

### Dataset
Telco Customer Churn Dataset — Kaggle  
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Technologies
- Python
- Pandas
- Scikit-learn
- Joblib

### ML Pipeline
1. Data ingestion and inspection.
2. Data cleaning and handling missing values.
3. Train-test split with stratification.
4. Categorical encoding using OneHotEncoder.
5. Numerical scaling using StandardScaler.
6. Classification using Logistic Regression.
7. Model evaluation.
8. Model persistence using Joblib.
9. Customer churn prediction using a reusable `predict()` function.

### Baseline Evaluation

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| ROC-AUC | 84.21% |

### Model Artifact
`models/churn_pipeline.joblib`

### Prediction Service
A reusable `predict()` helper accepts customer information and returns:
- Predicted churn status (`Yes` or `No`).
- Estimated churn probability.

### Future Development
This baseline serves as the starting point for future improvements to RetailMind AI.