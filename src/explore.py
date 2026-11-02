import pandas as pd

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print(df.shape)
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Check churn percentages
print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Check blank values in TotalCharges
print("\nBlank TotalCharges:")
print(df["TotalCharges"].str.strip().eq("").sum())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Inspect rows with missing TotalCharges
print("\nRows with missing TotalCharges:")

print(
    df[df["TotalCharges"].isnull()]
    [["tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
)

# Fill missing TotalCharges with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nUpdated data types:")
print(df.dtypes)


# Separate features and target

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify numerical and categorical columns

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns

# Create preprocessing transformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Fit on training data and transform it
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)