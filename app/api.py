"""Servicio FastAPI para evaluación académica; NO para decisiones médicas."""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import FEATURES, prepare_features

app = FastAPI(
    title="Clasificador de Enfermedad Cardíaca — Edwin Yunis y Jairo Serrano",
    description="Demostración académica de MLOps; no constituye un dispositivo médico.",
    version="1.0.0",
)


class Patient(BaseModel):
    Age: int = Field(ge=1, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: float = Field(ge=0, le=350)
    Cholesterol: float = Field(ge=0, le=1000)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=20, le=250)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=-5, le=10)
    ST_Slope: Literal["Up", "Flat", "Down"]


@lru_cache(maxsize=1)
def get_model():
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    if not model_path.exists():
        raise FileNotFoundError("Falta el modelo; ejecute python -m scripts.train")
    return joblib.load(model_path)


@app.get("/health")
def health():
    try:
        get_model()
    except (OSError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail="Modelo no disponible") from exc
    return {"status": "ready"}


@app.get("/model-info")
def model_info():
    return {
        "autores": ["Edwin Yunis", "Jairo Serrano"],
        "features": FEATURES,
        "target": "HeartDisease",
        "uso": "Demostración académica; no diagnóstico ni riesgo clínico validado.",
    }


@app.post("/predict")
def predict(patient: Patient):
    try:
        model = get_model()
        frame = prepare_features(pd.DataFrame([patient.model_dump()]))
        score = float(model.predict_proba(frame)[0, 1])
        return {
            "heart_disease_score": round(score, 6),
            "prediction": int(score >= 0.5),
            "threshold": 0.5,
            "disclaimer": "Puntuación académica no calibrada clínicamente.",
        }
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
