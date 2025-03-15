import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load necessary files
rf_model = joblib.load("RF_predictive_maintenance.pkl")  # Trained model
scaler_info = joblib.load("scaler.pkl")  # Scaler info
scaler = scaler_info["scaler"]  # Extract scaler
medians = joblib.load("medians.pkl")  # Load medians

# Define slider ranges based on the image
slider_ranges = {
    "Operational Hours": {"min": 5, "max": 170, "step": 0.1, "default": 96},
    "Process Temperature": {"min": 307, "max": 310, "step": 0.1, "default": 308.9},
    "Air Temperature": {"min": 296, "max": 300, "step": 0.1, "default": 298.8},
}

# Expected features in the model
expected_features = [
    "Air Temperature", "Process Temperature", "Rotational Speed", 
    "Torque", "Vibration Levels", "Operational Hours"
]

# Streamlit app title
st.title("PREDICTIVE MAINTENANCE SYSTEM")

# Create sliders for user input
input_values = {}
for feature, params in slider_ranges.items():
    input_values[feature] = st.slider(
        feature,
        min_value=float(params["min"]),
        max_value=float(params["max"]),
        value=float(params["default"]),
        step=float(params["step"]),
    )

# Construct DataFrame using sliders for certain features and medians for others
X_new = {
    "Air Temperature": input_values["Air Temperature"],
    "Process Temperature": input_values["Process Temperature"],
    "Rotational Speed": medians["Rotational Speed"],
    "Torque": medians["Torque"],
    "Vibration Levels": medians["Vibration Levels"],
    "Operational Hours": input_values["Operational Hours"]
}

# Convert to DataFrame ensuring column order is correct
X_new_df = pd.DataFrame([X_new], columns=expected_features)

# Standardize input data using the saved scaler
X_new_scaled = scaler.transform(X_new_df)

# Predict button
if st.button("Predict"):
    prediction = rf_model.predict(X_new_scaled)[0]

    if prediction == 0:
        st.markdown(
            '<div style="background-color:green;color:white;padding:20px;text-align:center;font-size:24px;border-radius:10px;">NO FAILURE PREDICTED</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background-color:red;color:white;padding:20px;text-align:center;font-size:24px;border-radius:10px;">FAILURE PREDICTED</div>',
            unsafe_allow_html=True,
        )
