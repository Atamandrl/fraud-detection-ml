# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Fraud Detection Demo")

# Model ve scaler yükle
data = joblib.load('models/lr_fe_smote_small.joblib')
model = data['model']
scaler = data['scaler']

st.write("Model expects:", model.n_features_in_)

# Kullanıcı inputları
amount_input = st.number_input("Transaction Amount", min_value=0.0, value=0.0, step=0.01)
time_input = st.number_input("Transaction Time", min_value=0.0, value=0.0, step=0.01)

if st.button("Predict"):

    # Input'u DataFrame olarak hazırla
    X_input = pd.DataFrame({
        'Time': [time_input],
        'LogAmount': [np.log1p(amount_input)],
        'Amt_by_Time': [amount_input / (time_input + 1)]
    })

    # Ölçekle
    X_input_scaled = scaler.transform(X_input)

    # Tahmin
    proba = model.predict_proba(X_input_scaled)[:,1]
    pred = (proba > 0.01).astype(int)  # threshold 0.01

    st.write("Prediction probability:", proba[0])
    st.write("Prediction (class, threshold 0.01):", pred[0])


