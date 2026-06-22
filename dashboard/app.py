"""
app.py  –  Streamlit dashboard for Bank Churn Prediction
Run:  streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Bank Churn Predictor", layout="wide")
st.title("🏦 Bank Customer Churn Prediction Dashboard")

@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

uploaded = st.sidebar.file_uploader("Upload customer CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
else:
    st.info("Upload a CSV file in the sidebar to get predictions.")
