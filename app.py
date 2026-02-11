import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Risk Prediction System")

# ----------- USER INPUTS -----------

age = st.number_input("Age", 20, 100)
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 400)
thalch = st.number_input("Maximum Heart Rate", 60, 220)
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0)
ca = st.number_input("Number of Major Vessels (0-3)", 0, 3)

sex = st.selectbox("Sex", ["Female", "Male"])
dataset = st.selectbox("Dataset Origin", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
cp = st.selectbox("Chest Pain Type", ["asymptomatic", "atypical angina", "non-anginal", "typical angina"])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
restecg = st.selectbox("Resting ECG", ["lv hypertrophy", "normal", "st-t abnormality"])
exang = st.selectbox("Exercise Induced Angina", ["False", "True"])
slope = st.selectbox("Slope", ["downsloping", "flat", "upsloping"])
thal = st.selectbox("Thal", ["fixed defect", "normal", "reversable defect"])

# ----------- PREDICTION -----------

if st.button("Predict Risk"):

    # Create base row with zeros
    input_dict = dict.fromkeys([
        'age','trestbps','chol','thalch','oldpeak','ca',
        'sex_Male',
        'dataset_Hungary','dataset_Switzerland','dataset_VA Long Beach',
        'cp_atypical angina','cp_non-anginal','cp_typical angina',
        'fbs_True',
        'restecg_normal','restecg_st-t abnormality',
        'exang_True',
        'slope_flat','slope_upsloping',
        'thal_normal','thal_reversable defect'
    ], 0)

    # Fill numeric values
    input_dict['age'] = age
    input_dict['trestbps'] = trestbps
    input_dict['chol'] = chol
    input_dict['thalch'] = thalch
    input_dict['oldpeak'] = oldpeak
    input_dict['ca'] = ca

    # Binary encodings
    if sex == "Male":
        input_dict['sex_Male'] = 1

    if dataset == "Hungary":
        input_dict['dataset_Hungary'] = 1
    elif dataset == "Switzerland":
        input_dict['dataset_Switzerland'] = 1
    elif dataset == "VA Long Beach":
        input_dict['dataset_VA Long Beach'] = 1

    if cp == "atypical angina":
        input_dict['cp_atypical angina'] = 1
    elif cp == "non-anginal":
        input_dict['cp_non-anginal'] = 1
    elif cp == "typical angina":
        input_dict['cp_typical angina'] = 1

    if fbs == "True":
        input_dict['fbs_True'] = 1

    if restecg == "normal":
        input_dict['restecg_normal'] = 1
    elif restecg == "st-t abnormality":
        input_dict['restecg_st-t abnormality'] = 1

    if exang == "True":
        input_dict['exang_True'] = 1

    if slope == "flat":
        input_dict['slope_flat'] = 1
    elif slope == "upsloping":
        input_dict['slope_upsloping'] = 1

    if thal == "normal":
        input_dict['thal_normal'] = 1
    elif thal == "reversable defect":
        input_dict['thal_reversable defect'] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    st.write("Risk Probability:", round(prob*100, 2), "%")

    if prob > 0.5:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk")
