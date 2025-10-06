from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model.joblib"
COLUMNS_PATH = ROOT / "training_columns.json"

model = joblib.load(MODEL_PATH)
with open(COLUMNS_PATH, "r") as f:
    TRAIN_COLUMNS: List[str] = json.load(f)

CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

class Input(BaseModel):
    Age: int
    Sex: str                 
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int          
    RestingECG: str     
    MaxHR: int
    ExerciseAngina: str     
    Oldpeak: float
    ST_Slope: str          

app = FastAPI(title="Heart Disease API", version="1.0")

def preprocess_input(payload: dict) -> pd.DataFrame:
    """Replica el preprocesamiento del entrenamiento:
       - convierte a DataFrame,
       - one-hot de categóricas con drop_first,
       - reindexa al orden exacto TRAIN_COLUMNS.
    """
    df_raw = pd.DataFrame([payload])
    df_enc = pd.get_dummies(df_raw, columns=CATEGORICAL_COLS, drop_first=True)
    df_enc = df_enc.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    return df_enc

@app.get("/")
def root():
    return {"message": "OK - use /docs para probar el endpoint /predict"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: Input):
    X = preprocess_input(data.dict())
    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba > 0.5)
    return {"heart_disease_probability": round(proba, 4), "prediction": pred}
