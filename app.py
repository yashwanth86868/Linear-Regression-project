"""
Streamlit frontend for Salary Prediction using Linear Regression.
Run: streamlit run app.py
"""
from pathlib import Path
import pickle

import streamlit as st

BASE_DIR = Path(r"C:\Users\ybali\Downloads\Investment.csv").resolve().parent
MODEL_PATH = BASE_DIR / "linear_regression_model.pkl"

st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Salary Prediction")
st.write(
    "Estimate a salary from years of experience using a trained "
    "Simple Linear Regression model."
)

if not MODEL_PATH.exists():
    st.warning("The trained model file was not found.")
    st.code("python slr.py\nstreamlit run app.py", language="bash")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

years = st.number_input(
    "Years of experience",
    min_value=0.0,
    max_value=60.0,
    value=2.0,
    step=0.5,
)

if st.button("Predict Salary", type="primary", use_container_width=True):
    prediction = float(model.predict([[years]])[0])
    st.success(f"Estimated salary: {prediction:,.2f}")
    st.caption(
        "This is an estimate from the trained dataset, not a guaranteed offer "
        "or a real-world salary assessment."
    )

st.divider()
st.caption("Built with Python, Pandas, Matplotlib, Scikit-learn, and Streamlit.")