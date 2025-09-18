import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# Load model and explainer
model = joblib.load("heart_model.pkl")
explainer = shap.TreeExplainer(model)

st.title("Heart Failure Prediction with SHAP")

# Collect input
age = st.slider("Age", 20, 90, 50)
creatinine = st.number_input("Serum Creatinine", min_value=0.1, max_value=10.0, value=1.2)
sodium = st.number_input("Serum Sodium", min_value=100, max_value=150, value=138)
ejection_fraction = st.slider("Ejection Fraction", 10, 80, 40)
high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
anaemia = st.selectbox("Anaemia", [0, 1])

#  Match the exact feature names used in training
user_input = pd.DataFrame({
    "age": [age],
    "serum_creatinine": [creatinine],
    "serum_sodium": [sodium],
    "ejection_fraction": [ejection_fraction],
    "high_blood_pressure": [high_blood_pressure],
    "diabetes": [diabetes],
    "anaemia": [anaemia]
})

# Ensure input columns match model training features
expected_features = model.get_booster().feature_names
user_input = user_input.reindex(columns=expected_features, fill_value=0)

if st.button("Predict"):
    pred = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]
    st.write(f"Prediction: {'High Risk' if pred==1 else 'Low Risk'} (Probability: {prob:.2f})")

    # SHAP explainability
    shap_values = explainer.shap_values(user_input)
    st.write("### Feature Contributions (SHAP)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, user_input, plot_type="bar", show=False)
    st.pyplot(fig)
