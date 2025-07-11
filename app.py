import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("best_random_forest_model.joblib")

# Set page title
st.set_page_config(page_title="Veteran Risk Tool", layout="centered")
st.title("🇺🇸 Veteran High Risk Prediction Tool")
st.markdown("### 📝 Enter Veteran Profile Info")

# Input fields
year = st.number_input("📅 Year", min_value=2000, max_value=2030, value=2024)
suicide_rate = st.number_input("⚠️ Veteran Suicide Rate per 100,000", min_value=0.0, value=10.0)

# Dropdowns
age_group = st.selectbox("👤 Select Age Group", ["18–34", "35–54", "55–74", "75+"])
gender = st.selectbox("🧍 Select Gender", ["Male", "Female"])
method = st.selectbox("🔫 Suicide Method", [
    "Firearms", "Poisoning", "Suffocation", "Other/Low-Count Methods", "Other Suicide"
])
state = st.selectbox("📍 State of Death", sorted([
    'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
    'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
    'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
    'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]))

# Prepare full feature list (match your model's training columns exactly)
model_features = [
    'Year', 'Veteran Suicide Rate per 100,000',
    'Age Group_35-54', 'Age Group_55-74', 'Age Group_75+',
    'Gender_Male',
    'Method_ Firearms', 'Method_ Poisoning ', 'Method_ Suffocation ',
    'Method_ Other and low-count methods ', 'Method_ Other suicide ',
    'State of Death_Alaska', 'State of Death_Arizona', 'State of Death_Arkansas', 'State of Death_California',
    'State of Death_Colorado', 'State of Death_Connecticut', 'State of Death_Delaware', 'State of Death_Florida',
    'State of Death_Georgia', 'State of Death_Hawaii', 'State of Death_Idaho', 'State of Death_Illinois',
    'State of Death_Indiana', 'State of Death_Iowa', 'State of Death_Kansas', 'State of Death_Kentucky',
    'State of Death_Louisiana', 'State of Death_Maine', 'State of Death_Maryland', 'State of Death_Massachusetts',
    'State of Death_Michigan', 'State of Death_Minnesota', 'State of Death_Mississippi', 'State of Death_Missouri',
    'State of Death_Montana', 'State of Death_Nebraska', 'State of Death_Nevada', 'State of Death_New Hampshire',
    'State of Death_New Jersey', 'State of Death_New Mexico', 'State of Death_New York',
    'State of Death_North Carolina', 'State of Death_North Dakota', 'State of Death_Ohio',
    'State of Death_Oklahoma', 'State of Death_Oregon', 'State of Death_Pennsylvania',
    'State of Death_Rhode Island', 'State of Death_South Carolina', 'State of Death_South Dakota',
    'State of Death_Tennessee', 'State of Death_Texas', 'State of Death_Utah', 'State of Death_Vermont',
    'State of Death_Virginia', 'State of Death_Washington', 'State of Death_West Virginia',
    'State of Death_Wisconsin', 'State of Death_Wyoming'
]

# Initialize feature DataFrame
input_df = pd.DataFrame([[0] * len(model_features)], columns=model_features)

# Set numerical fields
input_df.at[0, 'Year'] = year
input_df.at[0, 'Veteran Suicide Rate per 100,000'] = suicide_rate

# One-hot encode user selections
if age_group == "35–54":
    input_df.at[0, 'Age Group_35-54'] = 1
elif age_group == "55–74":
    input_df.at[0, 'Age Group_55-74'] = 1
elif age_group == "75+":
    input_df.at[0, 'Age Group_75+'] = 1
# Note: Age group "18–34" will remain 0 for all

if gender == "Male":
    input_df.at[0, 'Gender_Male'] = 1

# Suicide method
method_map = {
    "Firearms": 'Method_ Firearms',
    "Poisoning": 'Method_ Poisoning ',
    "Suffocation": 'Method_ Suffocation ',
    "Other/Low-Count Methods": 'Method_ Other and low-count methods ',
    "Other Suicide": 'Method_ Other suicide '
}
if method in method_map:
    input_df.at[0, method_map[method]] = 1

# State of death
state_col = f"State of Death_{state}"
if state_col in input_df.columns:
    input_df.at[0, state_col] = 1

# Prediction
if st.button("🔍 Predict Risk"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.markdown(
                f"🛑 **High Risk Detected!**\n\n"
                f"This veteran profile has a **{probability:.0%} probability** of high suicide risk. "
                f"Please prioritize intervention and support.",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"✅ **Low Risk**\n\n"
                f"This profile has a **{probability:.0%} probability** of high risk. Continue monitoring.",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")


