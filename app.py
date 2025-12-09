import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Model ve scaler'ı yükle
data = joblib.load('lr_fe_smote_small.joblib')
model = data['model']
scaler = data['scaler']

st.title("Fraud Detection Demo")

# Kullanıcı inputları
amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0)
time = st.number_input("Transaction Time", min_value=0.0, value=0.0)

# Inputları dataframe olarak hazırla
X_input = pd.DataFrame({
    'Time': [time],
    'LogAmount': [np.log1p(amount)],
    'Amt_by_Time': [amount / (time + 1)]
})

# Scale
X_input[['Time','LogAmount','Amt_by_Time']] = scaler.transform(X_input[['Time','LogAmount','Amt_by_Time']])

# Tahmin
proba = model.predict_proba(X_input)[:,1]
pred_class = (proba >= 0.5).astype(int)

st.write("Tahmin (probability):", proba)
st.write("Tahmin (class, threshold 0.5):", pred_class)
