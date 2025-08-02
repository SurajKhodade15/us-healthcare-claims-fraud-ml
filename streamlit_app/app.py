import streamlit as st
import pandas as pd
import joblib
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessing import preprocess_input

# Load model
model = joblib.load('models/cat_boost_model.pkl')

st.title("Healthcare Fraud Detection Dashboard")

st.sidebar.header("Input Claim Data")

# Example inputs - adjust as per your features
patient_age = st.sidebar.number_input('Patient Age', min_value=0, max_value=120, value=30)
procedure_code = st.sidebar.text_input('Procedure Code', value='93610')
# Add more inputs...

# Collect inputs into DataFrame
input_data = pd.DataFrame({
    'Patient_Age': [patient_age],
    'Procedure_Code': [procedure_code],
    # Add other features accordingly
})

# Preprocess inputs
processed_data = preprocess_input(input_data)

if st.button("Predict Fraud Probability"):
    prediction_proba = model.predict_proba(processed_data)[:, 1][0]
    st.write(f"Fraud Probability: {prediction_proba:.2%}")
    st.write("Prediction:", "Fraud" if prediction_proba > 0.5 else "Not Fraud")
