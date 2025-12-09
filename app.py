# app.py
import streamlit as st
import numpy as np
import joblib

# Başlık
st.title("Fraud Detection Demo")
st.write("Bu demo, kredi kartı işlemlerinde fraud (sahte) tespiti yapmaktadır.")

# 1️⃣ Model ve scaler yükle
# models klasöründe 'lr_fe_smote_small.joblib' olmalı
data = joblib.load('models/lr_fe_smote_small.joblib')
model = data['model']
scaler = data['scaler']

# 2️⃣ Kullanıcı inputları
amount_input = st.number_input("Transaction Amount", min_value=0.0, value=0.0, step=0.01)
time_input = st.number_input("Transaction Time", min_value=0.0, value=0.0, step=0.01)

# 3️⃣ Tahmin butonu
if st.button("Predict"):

    # Input'u modelin beklediği 2D array formatına çevir
    X_input = np.array([[time_input, np.log1p(amount_input), amount_input / (time_input + 1)]])
    
    # Ölçekle
    X_input_scaled = scaler.transform(X_input)

    # Tahmin
    proba = model.predict_proba(X_input_scaled)[:,1]
    pred = (proba > 0.01).astype(int)  # threshold 0.01 ile fraud tespiti

    # Sonuçları göster
    st.write("Prediction probability:", proba[0])
    st.write("Prediction (class, threshold 0.01):", pred[0])
