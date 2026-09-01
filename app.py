"""
FastAPI service for real-time fraud scoring.

Run:
    uvicorn app:app --reload --port 8000

Test:
    POST /predict with JSON body of feature values (see /schema for field list)
"""
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, create_model

app = FastAPI(title="Fraud Detection API")

model = joblib.load("models/xgb_model.joblib")
explainer = joblib.load("models/shap_explainer.joblib")
threshold = joblib.load("models/best_threshold.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

# Dynamically build a pydantic model matching the training feature columns
fields = {col: (float, ...) for col in feature_columns}
Transaction = create_model("Transaction", **fields)


@app.get("/")
def root():
    return {"status": "ok", "model": "xgboost_fraud_v1"}


@app.get("/schema")
def schema():
    return {"features_required": feature_columns, "threshold": float(threshold)}


@app.post("/predict")
def predict(transaction: Transaction):
    row = pd.DataFrame([transaction.dict()])[feature_columns]
    prob = float(model.predict_proba(row)[:, 1][0])
    is_fraud = bool(prob >= threshold)

    # Top 3 contributing features via SHAP
    shap_vals = explainer.shap_values(row)[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_features = [
        {"feature": feature_columns[i], "shap_value": float(shap_vals[i])}
        for i in top_idx
    ]

    return {
        "fraud_probability": prob,
        "is_fraud": is_fraud,
        "threshold_used": float(threshold),
        "top_contributing_features": top_features,
    }