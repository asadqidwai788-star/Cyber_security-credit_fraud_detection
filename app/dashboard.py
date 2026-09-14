"""
dashboard.py
Streamlit UI: upload a CSV of transactions, see which ones are flagged as fraud.

Run:
    streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Credit Card Fraud Detection — Dashboard")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

uploaded_file = st.file_uploader("Upload a CSV of transactions", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} transactions.")

    expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns
    missing = set(expected_cols) - set(df.columns)

    if missing:
        st.error(f"Uploaded file is missing required columns: {missing}")
    else:
        X = df[expected_cols]
        df["fraud_prediction"] = model.predict(X)
        df["fraud_probability"] = model.predict_proba(X)[:, 1]

        fraud_count = int(df["fraud_prediction"].sum())
        st.metric("Flagged transactions", fraud_count)

        st.subheader("Flagged as Fraud")
        st.dataframe(df[df["fraud_prediction"] == 1].sort_values("fraud_probability", ascending=False))

        st.subheader("All Transactions")
        st.dataframe(df)
