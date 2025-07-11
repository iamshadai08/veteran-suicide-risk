import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_random_forest_model.joblib")

# Define full feature list expected by the model
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

# --- Streamlit UI ---
st.title("🇺🇸 Veteran High Risk Prediction Tool")
st.header("📝 Enter Veteran Profile Info")

year = st.number_input("📅 Year", min_value=2000, max_value=2030, value=2024)
suicide_rate = st.number_input("⚠️ Veteran Suicide Rate per 100,000", min_value=0.0, value=10.0)

age_group = st.selectbox("👤 Select Age Group", ["35–54", "55–74", "75+"])
gender = st.selectbox("🧍‍♂️ Select Gender", ["Male", "Female"])
method = st.selectbox("⚰️ Suicide Method", [
    "Other/Low-Count Methods",
    "Other Suicide",
    "Poisoning",
    "Suffocation"
])

# Extract state from list of feature names
states = sorted([col.replace("State of Death_", "") for col in model_features if col.startswith("State of Death_")])
state = st.selectbox("📍 State of Death", states)

# Create input row with all 0s
input_df = pd.DataFrame([[0]*len(model_features)], columns=model_features)
input_df.at[0, 'Year'] = year
input_df.at[0, 'Veteran Suicide Rate per 100,000'] = suicide_rate

# One-hot encode based on selections
if age_group == "35–54":
    input_df.at[0, 'Age Group_35-54'] = 1
elif age_group == "55–74":
    input_df.at[0, 'Age Group_55-74'] = 1
elif age_group == "75+":
    input_df.at[0, 'Age Group_75+'] = 1

if gender == "Male":
    input_df.at[0, 'Gender_Male'] = 1

if method == "Other/Low-Count Methods":
    input_df.at[0, 'Method_ Other and low-count methods '] = 1
elif method == "Other Suicide":
    input_df.at[0, 'Method_ Other suicide '] = 1
elif method == "Poisoning":
    input_df.at[0, 'Method_ Poisoning '] = 1
elif method == "Suffocation":
    input_df.at[0, 'Method_ Suffocation '] = 1

state_feature = f"State of Death_{state}"
if state_feature in input_df.columns:
    input_df.at[0, state_feature] = 1

# --- Prediction ---
if st.button("🔍 Predict Risk"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        st.success(f"Predicted High Risk: {prediction} (Probability: {probability:.2f})")
    except Exception as e:
        st.error(f"Model error: {e}")


