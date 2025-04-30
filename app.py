import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("insurance_model.pkl", "rb"))

# Streamlit App Title
st.title("🏥 Insurance Charges Predictor")
st.write("Enter your details to predict the estimated insurance charges.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sex = st.selectbox("Sex", ("Male", "Female"))
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ("No", "Yes"))
region = st.selectbox("Region", ("Northwest", "Northeast", "Southeast", "Southwest"))

# Convert categorical inputs to numerical
sex_val = 0 if sex == "Male" else 1
smoker_val = 1 if smoker == "Yes" else 0
region_dict = {"Northwest": 0, "Northeast": 1, "Southeast": 2, "Southwest": 3}
region_val = region_dict[region]

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
    prediction = model.predict(input_data)[0]
    st.success(f"💵 Estimated Insurance Charges: ${prediction:.2f}")
