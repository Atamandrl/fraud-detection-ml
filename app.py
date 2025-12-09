import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Modeli yükle
model_data = joblib.load('lr_fe_smote_small.joblib')
model = model_data['model']
scaler = model_data['scaler']

st.title("Fraud Detection Demo")

# Kullanıcıdan input al
amount = st.number_input("Transaction Amount", min_value=0.0)
time = st.number_input("Transaction Time", min_value=0.0)

if st.button("Predict"):
    X_input = pd.DataFrame([[time, amount]], columns=['Time', 'Amount'])
    # Feature engineering
    X_input['LogAmount'] = np.log1p(X_input['Amount'])
    X_input['Amt_by_Time'] = X_input['Amount'] / (X_input['Time'] + 1)
    X_input.drop(columns=['Amount'], inplace=True)
    # Scale
    X_input[['Time','LogAmount','Amt_by_Time']] = scaler.transform(X_input[['Time','LogAmount','Amt_by_Time']])
    # Tahmin
    proba = model.predict_proba(X_input)[:,1]
    st.write("Fraud probability:", proba[0])
