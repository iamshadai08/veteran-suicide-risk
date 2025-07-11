
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_random_forest_model.joblib")

st.title("Veteran High Risk Prediction Tool")

# Input form
st.header("Enter Veteran Profile Info")
year = st.number_input("Year", min_value=2000, max_value=2030, value=2024)
suicide_rate = st.number_input("Veteran Suicide Rate per 100,000", min_value=0.0)

# Simulated feature toggles
age_35_54 = st.checkbox("Age Group 35-54")
gender_male = st.checkbox("Male")
state_ca = st.checkbox("California")
method_firearm = st.checkbox("Firearm Method")

# Create input DataFrame (only partial features for demo)
input_df = pd.DataFrame([{
    "Year": year,
    "Veteran Suicide Rate per 100,000": suicide_rate,
    "Age Group_35-54": int(age_35_54),
    "Gender_Male": int(gender_male),
    "State of Death_California": int(state_ca),
    "Method_ Firearms": int(method_firearm)
}])

# Prediction
if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted High Risk: {prediction} (Probability: {probability:.2f})")
