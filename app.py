import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_random_forest_model.joblib")

st.title("🇺🇸 Veteran High Risk Prediction Tool")
st.header("📝 Enter Veteran Profile Info")

# User Inputs
year = st.number_input("📅 Year", min_value=2000, max_value=2030, value=2024)
suicide_rate = st.number_input("⚠️ Veteran Suicide Rate per 100,000", min_value=0.0, value=10.0)

# Age group options
age_group_options = {
    "18–34": "Age Group_18-34",
    "35–54": "Age Group_35-54",
    "55–74": "Age Group_55-74",
    "75+": "Age Group_75+"
}
age_group_label = st.selectbox("👤 Select Age Group", list(age_group_options.keys()))

# Gender
gender = st.selectbox("🧍 Select Gender", ["Male"])
gender_feature = "Gender_Male"

# Suicide Method
method_map = {
    "Firearms": 'Method_ Firearms',
    "Poisoning": 'Method_ Poisoning ',
    "Suffocation": 'Method_ Suffocation ',
    "Other/Low-Count Methods": 'Method_ Other and low-count methods ',
    "Other Suicide": 'Method_ Other suicide '
}
method_label = st.selectbox("🧠 Suicide Method", list(method_map.keys()))
method_feature = method_map[method_label]

# States
states = sorted([col.replace("State of Death_", "") for col in model.feature_names_in_ if "State of Death_" in col])
selected_state = st.selectbox("📍 State of Death", states)
state_feature = f"State of Death_{selected_state}"

# Define all model features
model_features = list(model.feature_names_in_)
input_df = pd.DataFrame([[0] * len(model_features)], columns=model_features)

# Fill in user inputs
input_df.at[0, 'Year'] = year
input_df.at[0, 'Veteran Suicide Rate per 100,000'] = suicide_rate
input_df.at[0, age_group_options[age_group_label]] = 1
input_df.at[0, gender_feature] = 1
input_df.at[0, method_feature] = 1
input_df.at[0, state_feature] = 1

# Predict button
if st.button("🔍 Predict Risk"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ This profile is classified as **HIGH RISK** with a probability of **{probability:.2%}**.")
        else:
            st.success(f"✅ This profile is classified as **LOW RISK** with a probability of **{probability:.2%}**.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


