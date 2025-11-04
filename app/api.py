from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model.joblib"
COLS_PATH = BASE_DIR / "training_columns.json"

try:
    with open(COLS_PATH) as f:
        training_columns = json.load(f)
except FileNotFoundError:
    print(f"ERROR: No se encuentra '{COLS_PATH}'. Ejecuta el notebook primero.")
    training_columns = []

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"ERROR: No se encuentra '{MODEL_PATH}'. Ejecuta el notebook primero.")
    model = None


app = FastAPI(
    title="Heart Disease Prediction API",
    description="API para predecir enfermedades cardíacas utilizando un modelo de Machine Learning",
    version="1.0"
)


class HeartData(BaseModel):
    Age: int = Field(..., example=54, description="Edad del paciente")
    Sex: str = Field(..., example="M", description="Sexo del paciente (M o F)")
    ChestPainType: str = Field(..., example="ATA", description="Tipo de dolor de pecho (ej. ATA, NAP, ASY, TA)")
    RestingBP: int = Field(..., example=140, description="Presión arterial en reposo")
    Cholesterol: int = Field(..., example=289, description="Colesterol sérico")
    FastingBS: int = Field(..., example=0, description="Azúcar en sangre en ayunas (1 > 120 mg/dl, 0 en otro caso)")
    RestingECG: str = Field(..., example="Normal", description="Resultados ECG en reposo (ej. Normal, ST, LVH)")
    MaxHR: int = Field(..., example=172, description="Frecuencia cardíaca máxima alcanzada")
    ExerciseAngina: str = Field(..., example="N", description="Angina inducida por ejercicio (Y o N)")
    Oldpeak: float = Field(..., example=1.0, description="Depresión del ST inducida por el ejercicio")
    ST_Slope: str = Field(..., example="Up", description="Pendiente del segmento ST (ej. Up, Flat, Down)")

    class Config:
        schema_extra = {
            "example": {
                "Age": 40,
                "Sex": "M",
                "ChestPainType": "ATA",
                "RestingBP": 140,
                "Cholesterol": 289,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 172,
                "ExerciseAngina": "N",
                "Oldpeak": 0.0,
                "ST_Slope": "Up"
            }
        }


def preprocess_input(data: dict, training_cols: list) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df_processed = pd.get_dummies(df, drop_first=True)
    df_final = df_processed.reindex(columns=training_cols, fill_value=0)
    return df_final


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Predicción de Enfermedades Cardíacas"}


@app.get("/health")
def health_check():
    if model and training_columns:
        return {"status": "OK", "message": "Modelo y columnas cargados."}
    else:
        return {"status": "ERROR", "message": "Modelo o columnas no encontrados."}


@app.post("/predict")
def predict(data: HeartData):
    
    if not model or not training_columns:
        return {"error": "Modelo no cargado. Revisa el estado de /health"}

    try:
        X = preprocess_input(data.model_dump(), training_columns)
    except Exception as e:
        return {"error": f"Error durante el preprocesamiento: {str(e)}"}
    
    try:
        prediction = model.predict(X)[0]
    except Exception as e:
        return {"error": f"Error durante la predicción: {str(e)}"}

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[0][1]
    else:
        probability = float(prediction) 

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }