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


## Day 2 — Model Comparison

### Objective
Compare Decision Tree, Random Forest and Support Vector Machine (SVM) models for customer churn prediction.

### Model Comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Decision Tree | 72.11% | 47.55% | 49.20% | 64.77% |
| Random Forest | 78.64% | 62.54% | 48.66% | 81.85% |
| SVM | 79.13% | 64.18% | 48.40% | 79.02% |

### Champion Model: Decision Tree

Decision Tree was selected as the Day 2 champion because it achieved the highest recall (49.20%) among the three models.

Although Random Forest and SVM achieved better accuracy, precision and ROC-AUC, the selection prioritised recall to identify as many potential churners as possible.

### Model Artifact
`models/churn_champion.joblib`

### Observation
The Day 1 Logistic Regression baseline outperformed all three Day 2 models on the evaluated metrics. This demonstrates that more complex algorithms do not necessarily guarantee better performance.