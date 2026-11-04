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

## Day 3 — Hyperparameter Tuning and Bias-Variance Analysis

### Objective
Tune Decision Tree, Random Forest and Support Vector Machine (SVM) models using GridSearchCV and evaluate them using 5-fold stratified cross-validation.

### Model Comparison

| Model | Training Recall | Cross-Validation Recall |
|---|---:|---:|
| Decision Tree | 55.17% | 51.37% |
| Random Forest | 60.84% | 51.77% |
| SVM | 53.34% | 53.38% |

### Champion Model: SVM

SVM was selected as the Day 3 champion because it achieved the highest cross-validation recall of 53.38%.

### Final Test Set Evaluation

| Metric | Score |
|---|---:|
| Accuracy | 78.78% |
| Precision | 61.76% |
| Recall | 52.67% |
| ROC-AUC | 82.46% |

### Bias-Variance Analysis

The learning curve shows that training and validation recall gradually converge as the training dataset increases.

The small difference between the final training and validation scores suggests limited overfitting. However, the relatively modest recall indicates potential underfitting or limitations in the current feature set.

Further feature engineering and model tuning may improve performance.

### Hyperparameter Tuning

GridSearchCV was used with 5-fold StratifiedKFold cross-validation. The models were tuned using recall as the scoring metric, with `"Yes"` designated as the positive class.

The best hyperparameters for each model were identified and printed during execution.

### Model Artifact

`models/churn_v3.joblib`

### Learning Curve

`models/churn_v3_learning_curve.png`


## Day 4 — Feature Engineering and Ensemble Learning

### Objective
Improve customer churn prediction through domain-specific feature engineering, PCA dimensionality reduction, and ensemble learning.

### Feature Engineering
The following features were added:
- Tenure groups: categorised customers based on subscription duration.
- Average monthly charge: calculated using total charges and tenure.
- Service count: calculated the number of subscribed services.
- Contract flags: identified month-to-month, one-year and two-year contracts.
- Customer flags: identified online security, technical support and dependents.

Total features after engineering: 28.

### Dimensionality Reduction
Applied Principal Component Analysis (PCA) to numerical features, retaining 95.45% of their explained variance.

### Ensemble Models
- Random Forest: Bagging ensemble.
- XGBoost: Boosting ensemble.

Both models were compared against the Day 3 SVM champion using the same train-test split.

### Day 4 Model Comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Day 3 SVM | 78.78% | 61.76% | 52.67% | 82.46% |
| Random Forest | 76.79% | 54.87% | 70.86% | 83.60% |
| XGBoost | 75.16% | 52.20% | 76.20% | 83.75% |

### Final Champion: XGBoost

XGBoost was selected as the final champion because it achieved the highest recall (76.20%) and ROC-AUC (83.75%) among the evaluated models.

Compared with the Day 3 SVM, recall improved by 23.53 percentage points.

The lower precision and accuracy represent a trade-off: the model identifies more potential churners but also produces more false-positive predictions.

### Cumulative Model Baseline

| Day | Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Day 1 | Logistic Regression | 80.55% | 65.72% | 55.88% | 84.21% |
| Day 2 | Decision Tree | 72.11% | 47.55% | 49.20% | 64.77% |
| Day 3 | SVM | 78.78% | 61.76% | 52.67% | 82.46% |
| Day 4 | XGBoost | 75.16% | 52.20% | 76.20% | 83.75% |

### Production Model Artifact

`models/churn_production.joblib`

The saved pipeline includes the fitted preprocessing steps and XGBoost classifier, allowing predictions on raw customer features without manually repeating preprocessing.

### Phase 1 Milestone
Completed the initial churn prediction pipeline, model comparison, hyperparameter tuning, feature engineering, dimensionality reduction and ensemble learning.
