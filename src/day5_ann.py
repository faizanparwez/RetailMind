
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------------------------
# 1. REPRODUCIBILITY
# --------------------------------------------------

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)


# --------------------------------------------------
# 2. LOAD AND CLEAN DATASET
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(
    "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
).fillna(0)


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# Same features used by Day 4
# --------------------------------------------------

print("Engineering features...")

X = df.drop(columns=["customerID", "Churn"]).copy()

y = df["Churn"].map({"No": 0, "Yes": 1})

X["tenure_group"] = pd.cut(
    X["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

X["average_monthly_charge"] = (
    X["TotalCharges"] / X["tenure"].replace(0, 1)
)

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

X["is_month_to_month"] = (
    X["Contract"] == "Month-to-month"
).astype(int)

X["is_one_year_contract"] = (
    X["Contract"] == "One year"
).astype(int)

X["is_two_year_contract"] = (
    X["Contract"] == "Two year"
).astype(int)

X["has_online_security"] = (
    X["OnlineSecurity"] == "Yes"
).astype(int)

X["has_tech_support"] = (
    X["TechSupport"] == "Yes"
).astype(int)

X["has_dependents"] = (
    X["Dependents"] == "Yes"
).astype(int)

print("Total engineered features:", X.shape[1])


# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# --------------------------------------------------
# 5. PREPROCESSING
# --------------------------------------------------

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

preprocessor = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        numerical_features
    ),
    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        categorical_features
    )
])

# Fit preprocessing on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

y_train_array = y_train.to_numpy(dtype=np.float32)
y_test_array = y_test.to_numpy(dtype=np.float32)

print("Processed training shape:", X_train_processed.shape)


# --------------------------------------------------
# 6. BUILD ANN MODEL
# --------------------------------------------------

print("\nBuilding ANN...")

ann = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(X_train_processed.shape[1],)
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

ann.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

ann.summary()


# --------------------------------------------------
# 7. HANDLE CLASS IMBALANCE
# --------------------------------------------------

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_array.astype(int)
)

class_weights = {
    int(label): float(weight)
    for label, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)


# --------------------------------------------------
# 8. EARLY STOPPING
# --------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# --------------------------------------------------
# 9. TRAIN ANN
# --------------------------------------------------

print("\nTraining ANN...")

history = ann.fit(
    X_train_processed,
    y_train_array,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weights,
    verbose=1
)

print("\nTraining completed!")
print("Epochs actually trained:", len(history.history["loss"]))


# --------------------------------------------------
# 10. ANN EVALUATION
# --------------------------------------------------

print("\nEvaluating ANN...")

ann_probabilities = ann.predict(
    X_test_processed,
    verbose=0
).ravel()

ann_predictions = (
    ann_probabilities >= 0.5
).astype(int)

ann_metrics = {
    "Accuracy": accuracy_score(
        y_test_array, ann_predictions
    ),
    "Precision": precision_score(
        y_test_array, ann_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test_array, ann_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test_array, ann_probabilities
    )
}


# --------------------------------------------------
# 11. LOAD DAY 4 XGBOOST CHAMPION
# --------------------------------------------------

print("\nLoading Day 4 XGBoost champion...")

xgboost_model = joblib.load(
    "../models/churn_production.joblib"
)

# The Day 4 pipeline expects the same engineered
# raw features created above.
xgb_predictions = xgboost_model.predict(X_test)

xgb_probabilities = xgboost_model.predict_proba(X_test)[:, 1]

xgb_metrics = {
    "Accuracy": accuracy_score(
        y_test_array, xgb_predictions
    ),
    "Precision": precision_score(
        y_test_array, xgb_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test_array, xgb_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test_array, xgb_probabilities
    )
}


# --------------------------------------------------
# 12. COMPARE ANN AGAINST XGBOOST
# --------------------------------------------------

comparison = pd.DataFrame(
    [xgb_metrics, ann_metrics],
    index=["Day 4 XGBoost", "Day 5 ANN"]
)

print("\nMODEL COMPARISON")
print("----------------")

print(
    comparison.to_string(
        formatters={
            metric: "{:.4f}".format
            for metric in comparison.columns
        }
    )
)


# --------------------------------------------------
# 13. SAVE ANN AND PREPROCESSOR
# --------------------------------------------------

os.makedirs("../models", exist_ok=True)

ann.save("../models/churn_ann.keras")

joblib.dump(
    preprocessor,
    "../models/churn_ann_preprocessor.joblib"
)

comparison.to_csv(
    "../models/day5_model_comparison.csv"
)

print("\nANN model saved successfully!")
print("Preprocessor saved successfully!")
print("Comparison saved successfully!")


# --------------------------------------------------
# 14. PLOT TRAINING HISTORY
# --------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "../models/churn_ann_learning_curve.png"
)

plt.show()

print("\nDay 5 completed successfully!")
