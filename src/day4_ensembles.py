
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Convert target into numeric labels:
# No = 0, Yes = 1
y = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop(columns=["customerID", "Churn"])


# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

print("\nEngineering new features...")

# Tenure groups
X["tenure_group"] = pd.cut(
    X["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

# Average monthly charge based on total charges
X["average_monthly_charge"] = (
    X["TotalCharges"] / X["tenure"].replace(0, 1)
)

# Number of services used by the customer
service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

X["service_count"] = (
    X[service_columns] == "Yes"
).sum(axis=1)

# Contract-related flags
X["is_month_to_month"] = (
    X["Contract"] == "Month-to-month"
).astype(int)

X["is_one_year_contract"] = (
    X["Contract"] == "One year"
).astype(int)

X["is_two_year_contract"] = (
    X["Contract"] == "Two year"
).astype(int)

# Additional service and customer flags
X["has_online_security"] = (
    X["OnlineSecurity"] == "Yes"
).astype(int)

X["has_tech_support"] = (
    X["TechSupport"] == "Yes"
).astype(int)

X["has_dependents"] = (
    X["Dependents"] == "Yes"
).astype(int)

print("Feature engineering completed!")
print("Total features after engineering:", X.shape[1])


# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# --------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns


# --------------------------------------------------
# 5. PREPROCESSING WITH PCA
# --------------------------------------------------

numeric_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numerical_features),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])


# --------------------------------------------------
# 6. LOAD DAY 3 CHAMPION FOR BASELINE COMPARISON
# --------------------------------------------------

print("\nEvaluating Day 3 champion...")

day3_model = joblib.load("../models/churn_v3.joblib")

# The Day 3 model expects the original features,
# without the newly engineered columns.
X_original = df.drop(columns=["customerID", "Churn"])

X_original_train, X_original_test, y_original_train, y_original_test = (
    train_test_split(
        X_original,
        df["Churn"],
        test_size=0.2,
        random_state=42,
        stratify=df["Churn"]
    )
)

day3_predictions = day3_model.predict(X_original_test)
day3_probabilities = day3_model.predict_proba(X_original_test)

# The positive class is "Yes".
yes_index = list(day3_model.classes_).index("Yes")

day3_probabilities = day3_probabilities[:, yes_index]

# Convert predictions to numeric labels for evaluation.
day3_predictions = (
    pd.Series(day3_predictions).map({"No": 0, "Yes": 1}).to_numpy()
)

day3_y_test = (
    y_original_test.map({"No": 0, "Yes": 1}).to_numpy()
)

results = []

results.append({
    "Model": "Day 3 SVM",
    "Accuracy": accuracy_score(day3_y_test, day3_predictions),
    "Precision": precision_score(day3_y_test, day3_predictions),
    "Recall": recall_score(day3_y_test, day3_predictions),
    "ROC-AUC": roc_auc_score(day3_y_test, day3_probabilities)
})


# --------------------------------------------------
# 7. DEFINE ENSEMBLE MODELS
# --------------------------------------------------

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,
        scale_pos_weight=(
            (y_train == 0).sum() / (y_train == 1).sum()
        ),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 8. TRAIN AND EVALUATE ENSEMBLES
# --------------------------------------------------

trained_models = {}

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("ROC-AUC:", round(auc, 4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": auc
    })

    trained_models[name] = pipeline


# --------------------------------------------------
# 9. COMPARE ALL MODELS
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nDAY 4 MODEL COMPARISON")
print("----------------------")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)

os.makedirs("../models", exist_ok=True)

results_df.to_csv(
    "../models/day4_model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 10. SELECT FINAL CHAMPION
# --------------------------------------------------

# Compare the new ensembles against the Day 3 baseline.
ensemble_results = results_df[
    results_df["Model"].isin(["Random Forest", "XGBoost"])
]

champion_name = ensemble_results.loc[
    ensemble_results["Recall"].idxmax(),
    "Model"
]

champion_model = trained_models[champion_name]

print("\nFinal Day 4 Champion:", champion_name)


# --------------------------------------------------
# 11. CHECK PCA INFORMATION
# --------------------------------------------------

pca_model = (
    champion_model
    .named_steps["preprocessor"]
    .named_transformers_["num"]
    .named_steps["pca"]
)

explained_variance = pca_model.explained_variance_ratio_.sum()

print(
    "\nPCA Explained Variance:",
    round(explained_variance * 100, 2),
    "%"
)


# --------------------------------------------------
# 12. SAVE PRODUCTION MODEL
# --------------------------------------------------

joblib.dump(
    champion_model,
    "../models/churn_production.joblib"
)

print("\nProduction model saved successfully!")
print("Location: models/churn_production.joblib")

print("\nDay 4 completed successfully!")
