import streamlit as st
import pickle
import numpy as np

st.title("🎓 Student Score Prediction App")

# Load trained model
@st.cache_resource
def load_model():
    with open("student_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.success("Model loaded successfully!")

# Input from user
st.header("Enter Study Hours")
hours = st.number_input("How many hours did you study?", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

# Predict button
if st.button("Predict Score"):
    prediction = model.predict(np.array([[hours]]))
    st.balloons()
    st.subheader(f"Predicted Score: {prediction[0]:.2f}%")
