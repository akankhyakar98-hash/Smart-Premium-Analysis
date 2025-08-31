import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and expected input columns
model = joblib.load("smartpremium_model.pkl")
model_input_columns = joblib.load("model_input_columns.pkl")

# Load scaler and list of scaled columns
scaler = joblib.load("smartpremium_scaler.pkl")
scaled_columns = joblib.load("scaled_columns.pkl")

# Sidebar inputs
st.sidebar.header("Enter Customer Details")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 100, 30)
    income = st.sidebar.number_input("Annual Income", min_value=10000, max_value=1000000, value=50000)
    dependents = st.sidebar.slider("Number of Dependents", 0, 10, 2)
    health_score = st.sidebar.slider("Health Score", 0, 100, 50)
    previous_claims = st.sidebar.slider("Previous Claims", 0, 10, 1)
    vehicle_age = st.sidebar.slider("Vehicle Age", 0, 20, 5)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
    insurance_duration = st.sidebar.slider("Insurance Duration (Years)", 1, 30, 5)
    policy_year = st.sidebar.slider("Policy Year", 2000, 2025, 2022)
    policy_month = st.sidebar.slider("Policy Month", 1, 12, 6)
    policy_weekday = st.sidebar.slider("Policy Weekday", 0, 6, 2)
    policy_age_days = st.sidebar.slider("Policy Age (Days)", 0, 5000, 1000)

    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.sidebar.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    smoking = st.sidebar.selectbox("Smoking Status", ["Yes", "No"])
    exercise = st.sidebar.selectbox("Exercise Frequency", ["Monthly", "Rarely", "Weekly", "Never"])
    property_type = st.sidebar.selectbox("Property Type", ["Condo", "House", "Apartment"])

    # Manual one-hot encoding
    data = {
        'Age': age,
        'Annual Income': income,
        'Number of Dependents': dependents,
        'Health Score': health_score,
        'Previous Claims': previous_claims,
        'Vehicle Age': vehicle_age,
        'Credit Score': credit_score,
        'Insurance Duration': insurance_duration,
        'Policy_Year': policy_year,
        'Policy_Month': policy_month,
        'Policy_Weekday': policy_weekday,
        'Policy_Age_Days': policy_age_days,

        'Gender_Male': 1 if gender == "Male" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
        'Gender_Other': 1 if gender == "Other" else 0,

        'Education Level_High School': 1 if education == "High School" else 0,
        "Education Level_Bachelor's": 1 if education == "Bachelor's" else 0,
        "Education Level_Master's": 1 if education == "Master's" else 0,
        'Education Level_PhD': 1 if education == "PhD" else 0,

        'Location_Urban': 1 if location == "Urban" else 0,
        'Location_Suburban': 1 if location == "Suburban" else 0,
        'Location_Rural': 1 if location == "Rural" else 0,

        'Policy Type_Basic': 1 if policy_type == "Basic" else 0,
        'Policy Type_Comprehensive': 1 if policy_type == "Comprehensive" else 0,
        'Policy Type_Premium': 1 if policy_type == "Premium" else 0,

        'Smoking Status_Yes': 1 if smoking == "Yes" else 0,
        'Smoking Status_No': 1 if smoking == "No" else 0,

        'Exercise Frequency_Monthly': 1 if exercise == "Monthly" else 0,
        'Exercise Frequency_Rarely': 1 if exercise == "Rarely" else 0,
        'Exercise Frequency_Weekly': 1 if exercise == "Weekly" else 0,
        'Exercise Frequency_Never': 1 if exercise == "Never" else 0,

        'Property Type_Condo': 1 if property_type == "Condo" else 0,
        'Property Type_House': 1 if property_type == "House" else 0,
        'Property Type_Apartment': 1 if property_type == "Apartment" else 0
    }

    return pd.DataFrame(data, index=[0])

# Get input
input_df = user_input_features()

# Align input with model columns
input_encoded = input_df.copy()
for col in model_input_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_input_columns]

# Apply scaling to numeric columns
columns_to_scale = [col for col in scaled_columns if col in input_encoded.columns]
input_encoded[columns_to_scale] = scaler.transform(input_encoded[columns_to_scale])

# App title and input preview
st.title("💰 SmartPremium: Insurance Cost Predictor")
st.subheader("Customer Profile")
st.dataframe(input_df)

# Prediction block
try:
    pred = model.predict(input_encoded)
    st.subheader("Predicted Insurance Premium")
    st.write(f"₹{pred[0]:,.2f}")
except Exception as e:
    st.error(f"Prediction failed: {e}")
