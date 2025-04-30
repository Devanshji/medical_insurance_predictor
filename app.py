import pandas as pd
import streamlit as st
import pickle

# Load model
model = pickle.load(open('insurancemodelf.pkl', 'rb'))

# Title
st.title("Medical Insurance Cost Predictor")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI")
children = st.number_input("Number of Children", min_value=0, max_value=10)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Predict button
if st.button("Predict Insurance Cost"):
    # Create a DataFrame with the correct column names
    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])

    # Make prediction
    prediction = model.predict(input_df)
    st.success(f"Predicted Insurance Cost: ${prediction[0]:.2f}")
