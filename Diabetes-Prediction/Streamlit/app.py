import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("model.pkl")

# Page Config
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# CSS Styling
st.markdown("""
<style>

.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🩺 About Project")
st.sidebar.info(
    """
    This Machine Learning project predicts
    whether a patient is diabetic or non-diabetic
    based on health parameters.
    """
)

# Title
st.markdown("""
<h1 style='text-align:center;color:#1E88E5;'>
🩺 Diabetes Prediction System
</h1>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;'>Enter patient details below</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# Patient Information
st.subheader("Patient Information")

col1,col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=100.0,
        value=25.0
    )

with col2:
    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "not current",
            "No Info",
            "ever"
        ]
    )

    hba1c = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=10.0,
        value=5.5
    )

    blood_glucose = st.number_input(
        "Blood Glucose Level",
        min_value=50,
        max_value=350,
        value=120
    )

# Predict Button
if st.button("🔍 Predict Diabetes"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [blood_glucose]
    })

    prediction = model.predict(input_data)[0]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk of Diabetes")
        st.warning("Please consult a healthcare professional.")
    else:
        st.success("Low Risk of Diabetes")
        st.info("Health indicators are normal.")

    try:
        probability = model.predict_proba(input_data)[0][1]

        st.metric(
            label="🩺 Diabetes Risk Score",
            value=f"{probability:.2%}"
        )

        st.progress(float(probability))

    except Exception:
        pass

# Footer
st.markdown("---")
st.caption("👨‍💻 Developed by Tirth Dabhi | Machine Learning Project")
