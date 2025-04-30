import streamlit as st
import pandas as pd
import numpy as np
from pickle import load

# Load trained model
@st.cache_resource
def load_model():
    return load(open('insurancemodelf.pkl', 'rb'))

model = load_model()

# App layout
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥")

# Title and description
st.title("🏥 Insurance Cost Predictor")
st.markdown("""
Predict your medical insurance charges based on personal factors.
Adjust the inputs below and click **Predict** to see the estimate.
""")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses machine learning to predict insurance costs based on:
    - Age
    - BMI
    - Number of children
    - Smoking status
    """)
    st.markdown("Model: XGBoost Regressor")

# Input section
st.header("Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 30)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0, step=0.1,
                   help="Body Mass Index (Normal range: 18.5-24.9)")

with col2:
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    smoker = st.radio("Smoking Status", ["No", "Yes"])

# Hidden fields that were dropped during training (for reference)
with st.expander("Additional Information (Not used in prediction)"):
    sex = st.radio("Sex", ["Male", "Female"], disabled=True,
                  help="This feature was not significant in our model")
    region = st.selectbox("Region", 
                         ["Northeast", "Northwest", "Southeast", "Southwest"],
                         disabled=True,
                         help="This feature was not significant in our model")

# Prediction button
if st.button("Predict Insurance Cost", type="primary"):
    # Prepare input (only using features the model needs)
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == "Yes" else 0]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display results
    st.balloons()
    st.success(f"## Estimated Insurance Charges: 💲{prediction[0]:,.2f}")
    
    # Add some interpretation
    if smoker == "Yes":
        st.warning("Smoking significantly increases insurance costs. Consider quitting to lower your premiums.")
    if bmi > 30:
        st.warning("Your BMI indicates obesity, which may affect your insurance rates. Maintaining a healthy weight can help reduce costs.")

# Add footer
st.markdown("---")
st.caption("Note: This is a predictive model and actual insurance quotes may vary.")
