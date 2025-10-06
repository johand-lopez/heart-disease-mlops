from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json

# ===============================================================
# Cargar modelo y columnas de entrenamiento
# ===============================================================

with open("training_columns.json") as f:
    training_columns = json.load(f)

model = joblib.load("model.joblib")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API para predecir enfermedades cardíacas utilizando un modelo de Machine Learning",
    version="1.0"
)


# ===============================================================
# Definición del esquema de entrada
# ===============================================================

class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# ===============================================================
# Funciones auxiliares
# ===============================================================

def preprocess_input(data: dict) -> pd.DataFrame:
    """Convierte los datos del usuario en un DataFrame compatible con el modelo."""
    df = pd.DataFrame([data])
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    return df[training_columns]


# ===============================================================
# Endpoints principales
# ===============================================================

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Predicción de Enfermedades Cardíacas"}


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.post("/predict")
def predict(data: HeartData):
    X = preprocess_input(data.model_dump())
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
