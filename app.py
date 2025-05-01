import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open('model1.pkl', 'rb'))

# App title
st.title("Insurance Charges Prediction App")

# Sidebar inputs
st.sidebar.header("Enter user input features")

age = st.sidebar.slider('Age', 18, 100, 30)
bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
children = st.sidebar.slider('Number of Children', 0, 5, 0)
smoker = st.sidebar.selectbox('Smoker?', ['yes', 'no'])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])  # not used
region = st.sidebar.selectbox('Region', ['northeast', 'southeast', 'southwest', 'northwest'])  # not used

# Convert smoker to 1/0
smoker_val = 1 if smoker == 'yes' else 0

# Prepare input
input_df = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker_val]
})

# Prediction
if st.button('Predict Charges'):
    prediction = model.predict(input_df)[0]
    st.success(f'Estimated Insurance Charges: ${prediction:.2f}')
