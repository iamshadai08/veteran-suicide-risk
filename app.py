
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

# Define all model features as columns
model_features = [
    'Year', 'Veteran Suicide Rate per 100,000',
    'Age Group_35-54', 'Age Group_55-74', 'Age Group_75+',
    'Gender_Male',
    'Method_ Firearms', 'Method_ Poisoning ', 'Method_ Suffocation ',
    'Method_ Other and low-count methods ', 'Method_ Other suicide ',
    'State of Death_California', 'State of Death_Alaska', 'State of Death_Arizona',
    'State of Death_Arkansas', 'State of Death_Colorado', 'State of Death_Connecticut',
    'State of Death_Delaware', 'State of Death_Florida', 'State of Death_Georgia',
    'State of Death_Hawaii', 'State of Death_Idaho', 'State of Death_Illinois',
    'State of Death_Indiana', 'State of Death_Iowa', 'State of Death_Kansas',
    'State of Death_Kentucky', 'State of Death_Louisiana', 'State of Death_Maine',
    'State of Death_Maryland', 'State of Death_Massachusetts', 'State of Death_Michigan',
    'State of Death_Minnesota', 'State of Death_Mississippi', 'State of Death_Missouri',
    'State of Death_Montana', 'State of Death_Nebraska', 'State of Death_Nevada',
    'State of Death_New Hampshire', 'State of Death_New Jersey', 'State of Death_New Mexico',
    'State of Death_New York', 'State of Death_North Carolina', 'State of Death_North Dakota',
    'State of Death_Ohio', 'State of Death_Oklahoma', 'State of Death_Oregon',
    'State of Death_Pennsylvania', 'State of Death_Rhode Island', 'State of Death_South Carolina',
    'State of Death_South Dakota', 'State of Death_Tennessee', 'State of Death_Texas',
    'State of Death_Utah', 'State of Death_Vermont', 'State of Death_Virginia',
    'State of Death_Washington', 'State of Death_West Virginia', 'State of Death_Wisconsin',
    'State of Death_Wyoming'
]

# Create a blank input row with all zeros
input_df = pd.DataFrame([[0] * len(model_features)], columns=model_features)

# Fill in actual user data
input_df.at[0, 'Year'] = year
input_df.at[0, 'Veteran Suicide Rate per 100,000'] = suicide_rate
input_df.at[0, 'Age Group_35-54'] = int(age_35_54)
input_df.at[0, 'Gender_Male'] = int(gender_male)
input_df.at[0, 'State of Death_California'] = int(state_ca)
input_df.at[0, 'Method_ Firearms'] = int(method_firearm)


# Prediction
if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted High Risk: {prediction} (Probability: {probability:.2f})")
