# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Başlık
st.title("Fraud Detection Demo")

# Model ve scaler yükle
model_path = os.path.join("models", "lr_fe_smote_small.joblib")
data = joblib.load(model_path)
model = data["model"]
scaler = data["scaler"]

st.write("Enter transaction details below:")

# Kullanıcı inputları
amount_input = st.number_input("Transaction Amount", min_value=0.0, value=0.0, step=0.01)
time_input = st.number_input("Transaction Time", min_value=0.0, value=0.0, step=0.01)

# Tahmin butonu
if st.button("Predict"):

    # Input'u modelin beklediği DataFrame formatına çevir
    X_input = pd.DataFrame({
        "Time": [time_input],
        "LogAmount": [np.log1p(amount_input)],
        "Amt_by_Time": [amount_input / (time_input + 1)]
    })

    # Ölçekle
    X_input_scaled = scaler.transform(X_input)

    # Tahmin
    proba = model.predict_proba(X_input_scaled)[:, 1]
    pred = (proba > 0.01).astype(int)  # düşük threshold ile fraud yakalamayı artırıyoruz

    # Sonuçları göster
    st.write(f"Prediction probability: {proba[0]:.4f}")
    st.write(f"Prediction (class, threshold 0.01): {pred[0]}")
