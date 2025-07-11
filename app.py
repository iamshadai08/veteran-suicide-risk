import streamlit as st
import pandas as pd
import joblib
import time

# Load model
model = joblib.load("best_random_forest_model.joblib")

# Feature columns used in training (no Age Group_18-34 or Method_Firearms if they weren’t in training)
model_features = [
    'Year', 'Veteran Suicide Rate per 100,000',
    'Age Group_35-54', 'Age Group_55-74', 'Age Group_75+',
    'Gender_Male',
    'Method_ Other and low-count methods ', 'Method_ Other suicide ',
    'Method_ Poisoning ', 'Method_ Suffocation ',
    'State of Death_Alaska', 'State of Death_Arizona', 'State of Death_Arkansas',
    'State of Death_California', 'State of Death_Colorado', 'State of Death_Connecticut',
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

# UI
st.markdown("🇺🇸 # Veteran High Risk Prediction Tool")
st.markdown("📝 ## Enter Veteran Profile Info")

year = st.number_input("📅 Year", min_value=2000, max_value=2030, value=2024)
suicide_rate = st.number_input("⚠️ Veteran Suicide Rate per 100,000", min_value=0.0)

age_group = st.selectbox("👤 Select Age Group", ["35–54", "55–74", "75+"])
gender = st.selectbox("🧍 Select Gender", ["Male", "Female"])
method = st.selectbox("🧠 Suicide Method", ["Other/Low-Count Methods", "Other Suicide", "Poisoning", "Suffocation"])
state = st.selectbox("📍 State of Death", sorted([col.split('_')[-1].strip() for col in model_features if col.startswith('State of Death_')]))

# Encode inputs
input_df = pd.DataFrame([[0] * len(model_features)], columns=model_features)
input_df.at[0, 'Year'] = year
input_df.at[0, 'Veteran Suicide Rate per 100,000'] = suicide_rate

# Age
if age_group == "35–54":
    input_df.at[0, 'Age Group_35-54'] = 1
elif age_group == "55–74":
    input_df.at[0, 'Age Group_55-74'] = 1
elif age_group == "75+":
    input_df.at[0, 'Age Group_75+'] = 1

# Gender
input_df.at[0, 'Gender_Male'] = 1 if gender == "Male" else 0

# Method
method_map = {
    "Other/Low-Count Methods": 'Method_ Other and low-count methods ',
    "Other Suicide": 'Method_ Other suicide ',
    "Poisoning": 'Method_ Poisoning ',
    "Suffocation": 'Method_ Suffocation '
}
if method_map[method] in input_df.columns:
    input_df.at[0, method_map[method]] = 1

# State
state_col = f"State of Death_{state}"
if state_col in input_df.columns:
    input_df.at[0, state_col] = 1

# Predict
if st.button("🔍 Predict Risk"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **This profile is classified as HIGH RISK with a probability of {probability * 100:.2f}%.**")
            st.markdown(
                "<div style='animation: blinker 1s linear infinite; color:red; font-weight:bold;'>"
                "🚨 HIGH RISK - PLEASE TAKE ACTION 🚨</div>",
                unsafe_allow_html=True,
            )
            st.markdown("### 📞 Resources and Suggestions")
            st.markdown("- [Veterans Crisis Line](https://www.veteranscrisisline.net/) — Call 988 then Press 1")
            st.markdown("- Talk to your VA provider")
            st.markdown("- Reach out to family or support services")
            st.markdown("- Join a local or online support group")
        else:
            st.success(f"✅ This profile is classified as LOW RISK with a probability of {probability * 100:.2f}%.")

        st.markdown("---")
        st.markdown("📋 **Model Disclaimer**")
        st.info(
            "This is a predictive model based on historical data. "
            "It is not a substitute for clinical judgment. If you or someone you know is at risk, please seek help immediately."
        )

        st.markdown("🗳️ **Suggestion Box**")
        st.text_area("What could make this tool better?")

    except Exception as e:
        st.error(f"An error occurred: {e}")
