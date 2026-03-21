import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config("Power Energy Prediction")
st.title("Power Energy Prediction Project")
st.subheader("By Vaishnavi Badade")

# Load preprocessor
pre = joblib.load("PowerPlant_model_pre.joblib")

# Rebuild the exact same model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Load weights (version-independent)
model.load_weights("PowerPlant_model_weights.weights.h5")

AT = st.number_input("Ambient Temperature")
V  = st.number_input("Exhaust Vacuum")
AP = st.number_input("Ambient Pressure")
RH = st.number_input("Relative Humidity")

submit = st.button("Predict Power Output")

if submit:
    data = {'AT': [AT], 'V': [V], 'AP': [AP], 'RH': [RH]}
    xnew = pd.DataFrame(data)
    xnew_pre = pre.transform(xnew)
    preds = model.predict(xnew_pre)
    st.success(f"⚡ Predicted Power Output: {preds[0][0]:.2f} MW")
