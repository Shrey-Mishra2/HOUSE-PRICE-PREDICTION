import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title("House Price Predictor")

# Input fields
st.header("Enter House Features")
med_inc = st.number_input("Median Income (in tens of thousands)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)
house_age = st.number_input("House Age (years)", min_value=0, max_value=100, value=30, step=1)
ave_rooms = st.number_input("Average Number of Rooms", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

# Prediction button
if st.button("Predict Price"):
    input_data = np.array([[med_inc, house_age, ave_rooms]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${prediction*100000:.2f}")  # Convert to dollars

# Add some info
st.info("Note: Prices are in dollars and based on the California Housing dataset.")
