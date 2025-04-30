import streamlit as st
import pandas as pd
from pickle import load

# Load the trained model
model = load(open('insurancemodelf.pkl', 'rb'))

# Set up the app
st.title("Insurance Cost Predictor")
st.write("Enter your details to estimate your insurance cost:")

# Create input fields
age = st.slider("Your Age", 18, 100, 30)
sex = st.radio("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.radio("Do you smoke?", ["No", "Yes"])
region = st.selectbox("Your Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# When user clicks the predict button
if st.button("Predict Cost"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex.lower()],  # Convert to lowercase
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker.lower()],  # Convert to lowercase
        'region': [region.lower()]  # Convert to lowercase
    })
    
    # Convert categories to numbers (same as training)
    input_data['sex'] = input_data['sex'].map({'male': 0, 'female': 1})
    input_data['smoker'] = input_data['smoker'].map({'yes': 1, 'no': 0})
    input_data['region'] = input_data['region'].map({
        'northwest': 0,
        'northeast': 1,
        'southeast': 2,
        'southwest': 3
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Show result
    st.success(f"Estimated Insurance Cost: ${prediction[0]:,.2f}")
    
    # Simple explanation
    if smoker == "Yes":
        st.warning("Smoking increases your insurance cost")
    if bmi > 30:
        st.warning("Higher BMI may increase your cost")

# Simple footer
st.write("Note: This is an estimate only")
