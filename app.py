import streamlit as st
import joblib
import numpy as np

# Pipeline yükle
saved = joblib.load('xgb_pipeline_fraud.joblib')
model = saved['model']
scaler = saved['scaler']

st.title("Credit Card Fraud Detection Demo")

st.write("Lütfen aşağıdaki alanları doldurun:")

# Kullanıcı inputları
time = st.number_input("Time", min_value=0.0, value=1000.0)
amount = st.number_input("Transaction Amount", min_value=0.0, value=50.0)

# Feature engineering
log_amount = np.log1p(amount)
amt_by_time = amount / (time + 1)

# Scale
scaled_features = scaler.transform([[time, log_amount, amt_by_time]])

# Predict
prob = model.predict_proba(scaled_features)[:,1][0]
threshold = 0.01
pred_class = int(prob > threshold)

st.write(f"Fraud probability: {prob:.6f}")
st.write(f"Fraud prediction (threshold={threshold}): {pred_class}")
%%writefile app.py
import streamlit as st
import joblib
import numpy as np

# Pipeline yükle
saved = joblib.load('xgb_pipeline_fraud.joblib')
model = saved['model']
scaler = saved['scaler']

st.title("Credit Card Fraud Detection Demo")

st.write("Lütfen aşağıdaki alanları doldurun:")

# Kullanıcı inputları
time = st.number_input("Time", min_value=0.0, value=1000.0)
amount = st.number_input("Transaction Amount", min_value=0.0, value=50.0)

# Feature engineering
log_amount = np.log1p(amount)
amt_by_time = amount / (time + 1)

# Scale
scaled_features = scaler.transform([[time, log_amount, amt_by_time]])

# Predict
prob = model.predict_proba(scaled_features)[:,1][0]
threshold = 0.01
pred_class = int(prob > threshold)

st.write(f"Fraud probability: {prob:.6f}")
st.write(f"Fraud prediction (threshold={threshold}): {pred_class}")
