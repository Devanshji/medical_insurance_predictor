import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

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

# Initialize label encoders for categorical columns
sex_encoder = LabelEncoder()
smoker_encoder = LabelEncoder()
region_encoder = LabelEncoder()

# Train the encoder on the unique values that you expect to encounter.
sex_encoder.fit(["male", "female"])
smoker_encoder.fit(["yes", "no"])
region_encoder.fit(["southeast", "southwest", "northeast", "northwest"])

# Prediction logic on button click
if st.button("Predict Insurance Cost"):
    # Encode categorical variables
    encoded_sex = sex_encoder.transform([sex])[0]
    encoded_smoker = smoker_encoder.transform([smoker])[0]
    encoded_region = region_encoder.transform([region])[0]

    # Create DataFrame for prediction with encoded variables
    input_df = pd.DataFrame([{
        'age': age,
        'sex': encoded_sex,
        'bmi': bmi,
        'children': children,
        'smoker': encoded_smoker,
        'region': encoded_region
    }])

    # Make prediction
    prediction = model.predict(input_df)
    st.success(f"Predicted Insurance Cost: ${prediction[0]:.2f}")
