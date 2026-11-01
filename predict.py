"""
Predict churn for customers.

Usage:
  python predict.py                      -> predicts for a built-in sample customer
  python predict.py data/new_customers.csv  -> predicts for every row in a CSV
                                              (same columns as training data, no 'Churn' needed)
"""
import os
import sys
import joblib
import pandas as pd

os.makedirs("outputs", exist_ok=True)
model = joblib.load("models/churn_model.joblib")
EXPECTED = list(model.named_steps["prep"].feature_names_in_)


def align(df):
    """Add any column the model was trained on but is missing here (real Kaggle data has extra ones)."""
    for col in EXPECTED:
        if col not in df.columns:
            df[col] = "No"          # safe default for yes/no service columns
    return df[EXPECTED]


if len(sys.argv) > 1:
    df = pd.read_csv(sys.argv[1])
    ids = df["customerID"] if "customerID" in df.columns else pd.Series(range(len(df)))
    X = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns])
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
    out = pd.DataFrame({"customer": ids, "churn_probability": model.predict_proba(align(X))[:, 1].round(3)})
    out["prediction"] = out["churn_probability"].apply(lambda p: "WILL CHURN" if p >= 0.5 else "Will stay")
    out.to_csv("outputs/predictions.csv", index=False)
    print(out.head(15).to_string(index=False))
    print("\nFull results saved to outputs/predictions.csv")
else:
    sample = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 290.0,
    }])
    p = model.predict_proba(align(sample))[0, 1]
    print("Sample customer (new, month-to-month, fiber, e-check)")
    print(f"Churn probability: {p:.1%}  ->  {'WILL CHURN' if p >= 0.5 else 'Will stay'}")
