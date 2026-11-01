"""
Generate a synthetic telecom customer dataset (same columns as the popular
Kaggle "Telco Customer Churn" dataset) so the project runs without downloads.

Usage:  python generate_data.py
Output: data/telco_churn.csv

TIP: If you want to use the REAL dataset, download "WA_Fn-UseC_-Telco-Customer-Churn.csv"
from Kaggle, rename it to telco_churn.csv and put it in the data/ folder.
train.py works with both.
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 7000

gender = rng.choice(["Male", "Female"], N)
senior = rng.choice([0, 1], N, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], N)
dependents = rng.choice(["Yes", "No"], N, p=[0.3, 0.7])
tenure = rng.integers(0, 73, N)
phone = rng.choice(["Yes", "No"], N, p=[0.9, 0.1])
multi = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], N))
internet = rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20])

def addon(p_yes=0.4):
    vals = rng.choice(["Yes", "No"], N, p=[p_yes, 1 - p_yes])
    return np.where(internet == "No", "No internet service", vals)

online_sec = addon(0.3)
tech_support = addon(0.3)
streaming_tv = addon(0.45)
contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
paperless = rng.choice(["Yes", "No"], N, p=[0.6, 0.4])
payment = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21],
)

monthly = 20 + (internet == "DSL") * 25 + (internet == "Fiber optic") * 55
monthly = monthly + (online_sec == "Yes") * 5 + (tech_support == "Yes") * 5 + (streaming_tv == "Yes") * 10
monthly = monthly + (multi == "Yes") * 5 + rng.normal(0, 3, N)
monthly = np.round(np.clip(monthly, 18, 120), 2)
total = np.round(monthly * np.maximum(tenure, 1) + rng.normal(0, 20, N), 2)
total = np.clip(total, 0, None)

# Hidden "true" churn rule (log-odds) -> realistic patterns for the model to learn
z = -1.0
z += (contract == "Month-to-month") * 1.4 - (contract == "Two year") * 1.3
z += (internet == "Fiber optic") * 0.8
z += (payment == "Electronic check") * 0.6
z += (online_sec == "No") * 0.4 + (tech_support == "No") * 0.4
z += senior * 0.3 + (paperless == "Yes") * 0.25
z -= (tenure / 72) * 2.8
z += (monthly - 65) / 60
z += rng.normal(0, 0.6, N)
churn = rng.random(N) < 1 / (1 + np.exp(-z))

df = pd.DataFrame({
    "customerID": [f"C{100000 + i}" for i in range(N)],
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
    "OnlineSecurity": online_sec, "TechSupport": tech_support, "StreamingTV": streaming_tv,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
    "Churn": np.where(churn, "Yes", "No"),
})
os.makedirs("data", exist_ok=True)
df.to_csv("data/telco_churn.csv", index=False)
print(f"Saved data/telco_churn.csv  | rows: {len(df)} | churn rate: {churn.mean():.1%}")
