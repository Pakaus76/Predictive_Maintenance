import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Cargar los archivos necesarios
rf_model = joblib.load("RF_predictive_maintenance.pkl")  # Modelo entrenado
scaler_info = joblib.load("scaler.pkl")  # Información del scaler
scaler = scaler_info["scaler"]  # Extraer el scaler
medians = joblib.load("medians.pkl")  # Cargar las medianas

# Definir los valores de los sliders basados en la imagen
slider_ranges = {
    "Operational Hours": {"min": 5, "max": 170, "step": 0.1, "default": 74.6},
    "Process Temperature": {"min": 307, "max": 310, "step": 0.1, "default": 308.1},
    "Air Temperature": {"min": 296, "max": 300, "step": 0.1, "default": 297.4},
}

# Características esperadas por el modelo
expected_features = [
    "Air Temperature", "Process Temperature", "Rotational Speed", 
    "Torque", "Vibration Levels", "Operational Hours"
]

# Título de la aplicación
st.title("PREDICTIVE MAINTENANCE SYSTEM")

# Contenedores para sliders
input_values = {}
for feature, params in slider_ranges.items():
    input_values[feature] = st.slider(
        feature,
        min_value=float(params["min"]),
        max_value=float(params["max"]),
        value=float(params["default"]),
        step=float(params["step"]),
    )

# Construcción del DataFrame de entrada con medianas
X_new = {feature: input_values.get(feature, medians[feature]) for feature in expected_features}

# Convertir a DataFrame y asegurar el orden de las columnas
X_new_df = pd.DataFrame([X_new], columns=expected_features)

# Estandarizar los datos con el scaler cargado
X_new_scaled = scaler.transform(X_new_df)

# Botón para realizar la predicción
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
