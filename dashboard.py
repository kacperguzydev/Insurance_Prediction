import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px

# Paths
MODEL_PATH = os.path.join("models", "insurance_model.pkl")
DATA_DIR = "data"

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("🚗 ClaimVision: Insurance Claim Prediction Dashboard")

st.header("🔮 Predict Insurance Claim")

# Input form
with st.form("prediction_form"):
    age = st.slider("Driver Age", 18, 100, 30)
    sex = st.selectbox("Gender", ["female", "male"])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["non-smoker", "smoker"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    charges = st.slider("Estimated Charges", 1000.0, 70000.0, 5000.0)

    submit = st.form_submit_button("Predict")

# Make prediction when form submitted
if submit:
    # Convert categorical inputs to match model encoding
    sex_code = {"female": 0, "male": 1}[sex]
    smoker_code = {"non-smoker": 0, "smoker": 1}[smoker]
    region_code = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}[region]

    input_df = pd.DataFrame([[
        age, sex_code, bmi, children, smoker_code, region_code, charges
    ]], columns=["age", "sex", "bmi", "children", "smoker", "region", "charges"])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    outcome = "Claim" if prediction == 1 else "No Claim"
    st.subheader(f"✅ Prediction: {outcome}")
    st.write(f"Probability of claim: {probability:.2%}")

st.header("📊 Historical Claim KPIs")

# Claim distribution chart
claim_dist_csv = os.path.join(DATA_DIR, "kpis_claim_distribution.csv")
if os.path.exists(claim_dist_csv):
    df_claim_dist = pd.read_csv(claim_dist_csv)
    fig = px.bar(df_claim_dist, x="insuranceclaim", y="count", title="Claim Distribution (Yes vs No)")
    st.plotly_chart(fig)

# Average age by outcome chart
age_outcome_csv = os.path.join(DATA_DIR, "kpis_age_by_outcome.csv")
if os.path.exists(age_outcome_csv):
    df_age_outcome = pd.read_csv(age_outcome_csv)
    fig = px.bar(df_age_outcome, x="insuranceclaim", y="average_age", title="Average Driver Age by Claim Outcome")
    st.plotly_chart(fig)
