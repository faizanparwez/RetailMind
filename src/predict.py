import joblib
import pandas as pd

# Load the saved pipeline

model = joblib.load("../models/churn_pipeline.joblib")
def predict(customer):
    # Convert customer dictionary into DataFrame
    customer_df = pd.DataFrame([customer])

    # Predict churn
    prediction = model.predict(customer_df)[0]

    # Get churn probability
    probability = model.predict_proba(customer_df)[0][1]

    return {
        "prediction": prediction,
        "churn_probability": round(float(probability), 4)
    }


# Test the service
if __name__ == "__main__":

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.5,
        "TotalCharges": 111.0
    }

    result = predict(customer)

    print("\nCustomer Churn Prediction")
    print(result)