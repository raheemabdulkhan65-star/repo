import streamlit as st
import pickle
import os

st.set_page_config(page_title="Student Prediction", page_icon="🎓")

st.title("🎓 Student Prediction App")
st.write("Student prediction model information")

model_path = "student_model.pkl"

@st.cache_resource
def load_model():
    with open(model_path, "rb") as file:
        return pickle.load(file)

if not os.path.exists(model_path):
    st.error("Model file student_model.pkl not found.")
else:
    try:
        model = load_model()

        st.success("Model loaded successfully!")

        st.subheader("Model Information")
        st.write("Model type:", type(model).__name__)

        features = getattr(model, "feature_names_in_", None)
        feature_count = getattr(model, "n_features_in_", None)
        classes = getattr(model, "classes_", None)

        if features is not None:
            st.write("Input features:", list(features))

        if feature_count is not None:
            st.write("Number of input features:", int(feature_count))

        if classes is not None:
            st.write("Available classes:", list(classes))

        st.info(
            "Model details are shown above. "
            "We will configure prediction inputs after identifying the model."
        )

    except Exception as e:
        st.error("Could not load the model.")
        st.code(str(e))
