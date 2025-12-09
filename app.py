# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Fraud Detection Demo")

# 1) Modeli ve scaler'ı yükle
model_scaler = joblib.load("lr_fe_smote_small.joblib")
model = model_scaler['model']
scaler = model_scaler['scaler']

# 2) Kullanıcı girişleri
amount = st.number_input("Transaction Amount", value=100.0, min_value=0.0, step=1.0)
time = st.number_input("Transaction Time", value=50.0, min_value=0.0, step=1.0)

# 3) Girdi DataFrame'i oluştur
X_input = pd.DataFrame({
    'Time': [time],
    'LogAmount': [np.log1p(amount)],
    'Amt_by_Time': [amount / (time + 1)],
})

# 4) Eksik V1-V28 kolonlarını 0 ile doldur
for col in [f'V{i}' for i in range(1,29)]:
    X_input[col] = 0

# 5) Scale numeric features
X_input[['Time','LogAmount','Amt_by_Time']] = scaler.transform(X_input[['Time','LogAmount','Amt_by_Time']])

# 6) Tahmin
proba = model.predict_proba(X_input)[:,1]

# 7) Sonucu göster
st.subheader("Prediction")
st.write(f"Fraud probability: {proba[0]:.6f}")

# Opsiyonel: Class tahmini
threshold = 0.01  # İstersen değiştir
pred_class = (proba >= threshold).astype(int)
st.write(f"Predicted class (threshold {threshold}): {pred_class[0]}")
