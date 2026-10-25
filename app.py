import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Prediction Assistant")
st.write("Enter the patient's medical metrics below to predict diabetes risk.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=250.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=150.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)

with col2:
    insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=79.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

if st.button("Predict Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1] * 100
    
    st.subheader("Results:")
    if prediction[0] == 1:
        st.error(f"High Risk of Diabetes (Probability: {probability:.1f}%)")
    else:
        st.success(f"Low Risk of Diabetes (Probability: {probability:.1f}%)")
