import streamlit as st
import pandas as pd
import os
import sys
import joblib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessing import preprocess_input
# Define category options (tailor these to your data)
CATEGORICALS = {
    "Patient_Gender": ["Male", "Female", "Other"],
    "Admission_Type": ["Emergency", "Elective", "Urgent", "Trauma", "Newborn"],
    # Add all relevant categories as needed
}

MODEL_PATH = "models/cat_boost_model.pkl"

# Load model with caching
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.set_page_config(page_title="Healthcare Fraud Detection", layout="centered")
st.title("🏥 Healthcare Claim Fraud Detection")
st.markdown(
    "Enter claim details below to find out the probability of fraud.<br/>"
    "Powered by CatBoost, using engineered health claim features.",
    unsafe_allow_html=True,
)

with st.form("single_claim_form"):
    st.header("Enter Claim Details")

    col1, col2 = st.columns(2)

    patient_age = col1.number_input("Patient Age", min_value=0, max_value=120, value=30, help="Age of the patient")
    gender = col2.selectbox("Gender", CATEGORICALS["Patient_Gender"], help="Patient gender")
    admission_type = col1.selectbox("Admission Type", CATEGORICALS["Admission_Type"], help="Type of hospital admission")
    procedure_code = col2.text_input("Procedure Code", value='93610', help="Procedure code for the claim")

    # Add more claim fields here, using appropriate widgets (number_input, selectbox, etc.)
    # e.g. claim_amount = col1.number_input("Claim Amount", min_value=0.0, ...)

    # Simple validations
    errors = []
    if not procedure_code.strip():
        errors.append("Procedure Code is required.")
    # Add additional field validations as needed

    submit = st.form_submit_button("Predict", disabled=bool(errors))

    if errors:
        for err in errors:
            st.error(err)

    if submit and not errors:
        # Build input DataFrame with the correct column names/order
        input_df = pd.DataFrame([{
            "Patient_Age": patient_age,
            "Patient_Gender": gender,
            "Admission_Type": admission_type,
            "Procedure_Code": procedure_code,
            # include all required features in the same order as model training
        }])
        try:
            with st.spinner("Analyzing claim..."):
                processed = preprocess_input(input_df)
                proba = model.predict_proba(processed)[:, 1][0]
                label = "Fraud" if proba > 0.5 else "Not Fraud"
                color = "danger" if proba > 0.5 else "success"
                st.metric("Fraud Probability", f"{proba:.2%}", help="Model-probability this claim is fraudulent")
                if proba > 0.5:
                    st.markdown(f"<span style='color:red;font-size:1.3em'>⚠️ Prediction: <b>Fraud</b></span>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:green;font-size:1.3em'>✅ Prediction: <b>Not Fraud</b></span>",
                                unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Sidebar/help
st.sidebar.title("How to use")
st.sidebar.info(
    "Fill out the form and get a fraud prediction instantly.\n"
    "All user input is validated before prediction. Only one claim at a time is supported in this demo app."
)
st.sidebar.markdown("Created with ❤️ using Streamlit and CatBoost")
