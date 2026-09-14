"""
main.py
FastAPI backend serving the trained fraud detection model.

Run:
    uvicorn app.main:app --reload

Then POST a transaction to http://127.0.0.1:8000/predict
"""

import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"

app = FastAPI(title="Credit Card Fraud Detection API")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")
    Time: float
    Amount: float
    # V1-V28 supplied as extra fields; kept flexible so the schema matches
    # whatever columns the trained model actually expects.


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python src/train.py` first to generate models/best_model.pkl.",
        )

    df = pd.DataFrame([transaction.model_dump()])

    # Align columns to what the model expects
    expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    df = df[expected_cols]

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {"fraud": bool(pred), "fraud_probability": round(float(proba), 4)}
 