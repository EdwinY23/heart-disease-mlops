"""Carga exclusiva del dataset oficial de Kaggle indicado por el docente."""
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import pandas as pd
import numpy as np

DATASET_ORIGINAL = "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction"
TARGET = "HeartDisease"
NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURES = NUMERIC + CATEGORICAL


def load_data(path="data/heart.csv", download=True):
    """Solo carga los registros oficiales de Kaggle; no genera filas ni usa espejos ajenos."""
    path = Path(path)
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Falta el CSV oficial de Kaggle: {path}")
        try:
            request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=120) as response:
                archive_bytes = response.read()
            with ZipFile(BytesIO(archive_bytes)) as archive:
                matches = [n for n in archive.namelist() if Path(n).name.lower() == "heart.csv"]
                if len(matches) != 1:
                    raise ValueError("El archivo oficial no contiene un único heart.csv.")
                frame = pd.read_csv(BytesIO(archive.read(matches[0])))
            validate_schema(frame)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        except Exception as exc:
            raise RuntimeError(
                "No se descargó el CSV oficial de Kaggle. Descárgalo desde "
                f"{DATASET_ORIGINAL} y guarda heart.csv en {path}; "
                "no se usarán otras fuentes ni datos de reemplazo."
            ) from exc
    frame = pd.read_csv(path)
    validate_schema(frame)
    return frame


def validate_schema(frame):
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f"No coincide con Heart Failure Prediction (Kaggle): {missing}")
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("HeartDisease debe tener etiquetas 0/1 sin faltantes")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Se requieren las dos clases del dataset oficial")
    return True


def prepare_features(frame):
    """No agrega pacientes ni variables no observadas; imputación dentro del pipeline."""
    x = frame[FEATURES].copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x[["RestingBP", "Cholesterol"]] = x[["RestingBP", "Cholesterol"]].replace(0, np.nan)
    for col in CATEGORICAL:
        x[col] = x[col].astype("string").fillna("DESCONOCIDO").astype(str)
    return x


def describe_data(frame):
    return {
        "n_registros": int(len(frame)),
        "n_variables": int(len(FEATURES)),
        "positivos": int(frame[TARGET].sum()),
        "prevalencia_observada": float(frame[TARGET].mean()),
        "duplicados_completos": int(frame.duplicated().sum()),
        "restingbp_cero": int((frame["RestingBP"] == 0).sum()),
        "cholesterol_cero": int((frame["Cholesterol"] == 0).sum()),
        "faltantes_originales": {k: int(v) for k, v in frame.isna().sum().items()},
    }
