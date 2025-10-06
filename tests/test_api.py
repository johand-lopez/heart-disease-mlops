from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Bienvenido" in r.json().get("message")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "OK"


def test_predict_valid_input():
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 230,
        "fbs": 0,
        "restecg": 1,
        "thalach": 170,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200

    response = r.json()
    assert "prediction" in response
    assert "probability" in response
    assert 0 <= response["probability"] <= 1
