
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    learning_curve
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer
)
recall_scorer = make_scorer(recall_score, pos_label="Yes")


# 1. Load and clean dataset
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)


# 2. Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Identify numerical and categorical features
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns


# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# 6. Define candidate models and parameter grids
model_grids = {
    "Decision Tree": {
        "classifier": [DecisionTreeClassifier(random_state=42)],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 5]
    },

    "Random Forest": {
        "classifier": [RandomForestClassifier(random_state=42)],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 5]
    },

    "SVM": {
        "classifier": [SVC(probability=True, random_state=42)],
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale", "auto"]
    }
}


# 7. Configure cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 8. Perform GridSearchCV for each model
best_models = {}
comparison_results = []

for name, param_grid in model_grids.items():

    print(f"\nTuning {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier())
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=recall_scorer,
        cv=cv,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    best_index = grid_search.best_index_

    train_score = grid_search.cv_results_[
        "mean_train_score"
    ][best_index]

    validation_score = grid_search.best_score_

    comparison_results.append({
        "Model": name,
        "Best CV Recall": validation_score,
        "Train Recall": train_score,
        "Best Parameters": grid_search.best_params_
    })

    print("Best CV Recall:", round(validation_score, 4))
    print("Train Recall:", round(train_score, 4))
    print("Best Parameters:", grid_search.best_params_)


# 9. Display model comparison
results_df = pd.DataFrame(comparison_results)

print("\nTuned Model Comparison")
print("----------------------")

print(
    results_df[
        ["Model", "Train Recall", "Best CV Recall"]
    ].to_string(index=False)
)


# 10. Select champion based on cross-validation recall
champion_name = results_df.loc[
    results_df["Best CV Recall"].idxmax(),
    "Model"
]

champion_model = best_models[champion_name]

print("\nChampion Model:", champion_name)


# 11. Evaluate champion on untouched test data
y_pred = champion_model.predict(X_test)
y_prob = champion_model.predict_proba(X_test)[:, 1]

print("\nFinal Test Set Evaluation")
print("-------------------------")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(
    "Precision:",
    round(precision_score(y_test, y_pred, pos_label="Yes"), 4)
)
print(
    "Recall:",
    round(recall_score(y_test, y_pred, pos_label="Yes"), 4)
)
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))


# 12. Save champion model
joblib.dump(champion_model, "../models/churn_v3.joblib")

print("\nChampion model saved successfully!")


# 13. Generate learning curve for champion
print("\nGenerating learning curve...")

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=champion_model,
    X=X_train,
    y=y_train,
    cv=cv,
    scoring=recall_scorer,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)

plt.figure(figsize=(8, 5))

plt.plot(train_sizes, train_mean, marker="o", label="Training Recall")
plt.plot(train_sizes, validation_mean, marker="o", label="Validation Recall")

plt.title(f"Learning Curve - {champion_name}")
plt.xlabel("Number of Training Samples")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)

plt.savefig("../models/churn_v3_learning_curve.png")
plt.show()

print("Learning curve saved successfully!")
