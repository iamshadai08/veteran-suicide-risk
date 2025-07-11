import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_random_forest_model.joblib")

# --- Header
st.title("🧠 Veteran Suicide Risk Predictor")
st.markdown("""
This tool uses historical data to **predict whether a veteran profile may be at high risk of suicide**.  
🔺 **Disclaimer**: This is not a diagnostic tool. Always consult mental health professionals.  
""", unsafe_allow_html=True)

# --- Sidebar: Links, Contact, Suggestions
st.sidebar.title("ℹ️ App Info")
st.sidebar.markdown("""
- 📄 [Data Source](https://www.mentalhealth.va.gov/suicide_prevention/data.asp)  
- 🛠️ [GitHub Code](https://github.com/yourrepo)  
- 📬 Contact: support@veteransafety.org  
""")
st.sidebar.text_input("💡 Suggest a feature:", placeholder="E.g., Add support for regions...")

# --- User Inputs
st.header("📋 Enter Veteran Profile")

year = st.slider("Year", min_value=2000, max_value=2030, value=2024)
rate = st.number_input("Veteran Suicide Rate per 100,000", min_value=0.0)

# Age group
age_group = st.selectbox("Age Group", ["18-34", "35-54", "55-74", "75+"])

# Gender
gender = st.selectbox("Gender", ["Male", "Female"])
gender_male = int(gender == "Male")  # only one-hot encoded feature needed

# State dropdown
states = [s.replace("State of Death_", "") for s in model.feature_names_in_ if "State of Death_" in s]
state = st.selectbox("State of Death", sorted(states))

# Method dropdown
methods = [m.replace("Method_", "").strip() for m in model.feature_names_in_ if "Method_" in m]
method = st.selectbox("Method Used", sorted(methods))

# --- Create input row
input_data = pd.DataFrame([[0] * len(model.feature_names_in_)], columns=model.feature_names_in_)
input_data.at[0, 'Year'] = year
input_data.at[0, 'Veteran Suicide Rate per 100,000'] = rate
input_data.at[0, 'Gender_Male'] = gender_male

# Age dummies
for group in ["35-54", "55-74", "75+"]:
    if age_group == group:
        input_data.at[0, f"Age Group_{group}"] = 1

# Method
method_col = f"Method_{method}"
if method_col in input_data.columns:
    input_data.at[0, method_col] = 1

# State
state_col = f"State of Death_{state}"
if state_col in input_data.columns:
    input_data.at[0, state_col] = 1

# --- Predict
if st.button("🔎 Predict Suicide Risk"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("⚠️ HIGH RISK: This veteran profile is flagged as HIGH RISK.")
            st.markdown(f"""<div style='padding: 15px; background-color: #ffdddd; 
                            color: red; font-size: 20px; border: 2px solid red; 
                            border-radius: 5px; animation: pulse 1s infinite;'>
                            ⚠️ PLEASE TAKE ACTION IMMEDIATELY
                        </div>
                        <style>
                        @keyframes pulse {{
                            0% {{ box-shadow: 0 0 0px red; }}
                            50% {{ box-shadow: 0 0 20px red; }}
                            100% {{ box-shadow: 0 0 0px red; }}
                        }}
                        </style>""", unsafe_allow_html=True)
        else:
            st.success("🟢 LOW RISK: This profile is not flagged as high risk.")

        st.write(f"🔢 Risk Confidence: **{proba:.2%}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Resources
st.markdown("---")
st.markdown("### 🧩 Mental Health Resources")
st.markdown("""
- 📞 **Veterans Crisis Line**: Call 988 then Press 1
- 🌐 [VA Suicide Prevention Site](https://www.mentalhealth.va.gov/suicide_prevention/)
- 💬 Confidential 24/7 Chat: [Click Here](https://www.veteranscrisisline.net/get-help-now/chat/)
""")

# --- Footer
st.markdown("---")
st.caption("© 2024 Veteran Risk Tool. For educational & awareness purposes only.")
