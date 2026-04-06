from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_predict_single():
    r = client.post("/predict", json={"text": "good"})
    assert r.status_code == 200
    assert "label" in r.json()

def test_predict_batch():
    r = client.post("/predict_batch", json={"texts": ["good", "bad"]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2