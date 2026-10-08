"""Pruebas HTTP con pacientes reales y un clasificador aprendido de esos mismos datos."""
import json
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from app import api
from src.data import FEATURES
from src.modeling import split_data, train_pipeline

client = TestClient(api.app)


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["autores"] == ["Edwin Yunis", "Jairo Serrano"]


def test_predict_with_observed_row(real_heart, monkeypatch):
    x_train, _, y_train, _ = split_data(real_heart)
    trained = train_pipeline(x_train, y_train, LogisticRegression(max_iter=1000),
                             {"clf__C": [1]}, cv=3).best_estimator_
    monkeypatch.setattr(api, "get_model", lambda: trained)
    record = json.loads(real_heart[FEATURES].iloc[[0]].to_json(orient="records"))[0]
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    assert 0 <= response.json()["heart_disease_score"] <= 1
    assert response.json()["prediction"] in (0, 1)


def test_missing_field_using_observed_row(real_heart):
    record = json.loads(real_heart[FEATURES].iloc[[0]].to_json(orient="records"))[0]
    record.pop("Age")
    assert client.post("/predict", json=record).status_code == 422
