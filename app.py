"""
Simple web app for churn prediction.   Run:  streamlit run app.py
"""
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.caption("Enter customer details to estimate the probability that they will leave.")



@st.cache_resource(show_spinner="First start: training the model (takes about 1 minute)...")
def load_model():
    """Load the saved model; if it is missing or incompatible, train it from data/telco_churn.csv."""
    path = "models/churn_model.joblib"
    try:
        return joblib.load(path)
    except Exception:
        subprocess.run([sys.executable, "train.py"], check=True)
        return joblib.load(path)


model = load_model()
EXPECTED = list(model.named_steps["prep"].feature_names_in_)

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Has partner", ["Yes", "No"])
    dependents = st.selectbox("Has dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multi = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
with c2:
    online_sec = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    stv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 10.0, 150.0, 70.0)
    total = st.number_input("Total charges", 0.0, 10000.0, float(monthly * max(tenure, 1)))

if st.button("Predict churn", type="primary"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": online_sec, "TechSupport": tech, "StreamingTV": stv, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total}])
    for col in EXPECTED:            # columns only present in the real Kaggle data
        if col not in row.columns:
            row[col] = "No"
    p = model.predict_proba(row[EXPECTED])[0, 1]
    st.metric("Churn probability", f"{p:.1%}")
    st.progress(float(p))
    if p >= 0.5:
        st.error("⚠️ High risk: this customer is likely to churn. Consider a retention offer.")
    else:
        st.success("✅ Low risk: this customer is likely to stay.")
