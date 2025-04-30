import streamlit as st
import pandas as pd
from pickle import load

# Load the model
model = load(open('insurancemodelf.pkl', 'rb'))

# Initialize label encoders for categorical columns (same as in training)
sex_encoder = {'male': 0, 'female': 1}
smoker_encoder = {'yes': 1, 'no': 0}
region_encoder = {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}

# Set up Streamlit UI
st.title("Insurance Charges Prediction")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI")
children = st.number_input("Number of Children", min_value=0, max_value=10)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "northeast", "southeast", "southwest"])

# Prepare input data for prediction
if st.button("Predict Insurance Cost"):
    # Encode categorical values as per model encoding
    encoded_sex = sex_encoder[sex]
    encoded_smoker = smoker_encoder[smoker]
    encoded_region = region_encoder[region]

    # Create a DataFrame to match the expected input structure
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [encoded_smoker],
        'region': [encoded_region]
    })

    # Predict
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f"Predicted Insurance Charges: ${prediction[0]:.2f}")
