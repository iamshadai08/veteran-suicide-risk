import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_random_forest_model.joblib")

st.title("🇺🇸 Veteran High Risk Prediction Tool")
st.header("📋 Enter Veteran Profile Info")

# Basic Inputs
year = st.number_input("📅 Year", min_value=2000, max_value=2030, value=2024)
suicide_rate = st.number_input("⚠️ Veteran Suicide Rate per 100,000", min_value=0.0, value=0.0)

# Age Group Dropdown
age_group = st.selectbox("👤 Select Age Group", ["", "35–54", "55–74", "75+"])

# Gender Dropdown
gender = st.selectbox("🚻 Select Gender", ["", "Male", "Female"])

# Suicide Method Dropdown
suicide_method = st.selectbox(
    "💀 Suicide Method",
    ["", "Firearms", "Poisoning", "Suffocation", "Other and low-count methods", "Other suicide"]
)

# State Dropdown with Search
states = sorted([
    "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
])
state = st.selectbox("📍 State of Death", [""] + states)

# --- Model Columns ---
model_features = [
    'Year', 'Veteran Suicide Rate per 100,000',
    'Age Group_35-54', 'Age Group_55-74', 'Age Group_75+',
    'Gender_Male',
    'Method_ Firearms', 'Method_ Poisoning ', 'Method_ Suffocation ',
    'Method_ Other and low-count methods ', 'Method_ Other suicide '
] + [f"State of Death_{s}" for s in states]

# Create DataFrame with zeros
input_df = pd.DataFrame([[0]*len(model_features)], columns=model_features)
input_df.at[0, 'Year'] = year
input_df.at[0, 'Veteran Suicide Rate per 100,000'] = suicide_rate

# Map inputs to one-hot encoded fields
if age_group:
    input_df.at[0, f"Age Group_{age_group}"] = 1
if gender == "Male":
    input_df.at[0, "Gender_Male"] = 1
if suicide_method:
    method_col = f"Method_ {suicide_method}"
    if method_col in input_df.columns:
        input_df.at[0, method_col] = 1
if state:
    state_col = f"State of Death_{state}"
    if state_col in input_df.columns:
        input_df.at[0, state_col] = 1

# --- Predict ---
if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"✅ Predicted High Risk: {prediction} (Probability: {probability:.2f})")

