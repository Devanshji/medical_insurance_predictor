import streamlit as st
import pandas as pd
from pickle import load

# Load the trained model
model = load(open('insurancemodelf.pkl', 'rb'))

# Set up the app
st.title("Insurance Cost Predictor")
st.write("Enter your details to estimate your insurance cost:")

# Create input fields - ONLY for features the model actually uses
# (Based on your training code, these are age, bmi, children, smoker)
age = st.slider("Your Age", 18, 100, 30)
bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.radio("Do you smoke?", ["No", "Yes"])

# When user clicks the predict button
if st.button("Predict Cost"):
    # Prepare the input data with ONLY the features the model uses
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == "Yes" else 0]  # Already encoded as number
    })
    
    # Make sure column order matches what model expects
    input_data = input_data[model.feature_names_in_]
    
    # Make prediction
    try:
        prediction = model.predict(input_data)
        st.success(f"Estimated Insurance Cost: ${prediction[0]:,.2f}")
        
        # Simple explanation
        if smoker == "Yes":
            st.warning("Smoking increases your insurance cost")
        if bmi > 30:
            st.warning("Higher BMI may increase your cost")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Input data used:", input_data)
        st.write("Model expects:", model.feature_names_in_)

# Simple footer
st.write("Note: This is an estimate only")
