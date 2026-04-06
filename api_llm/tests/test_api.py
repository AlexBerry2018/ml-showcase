from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()

def test_generate_endpoint():
    payload = {"prompt": "Say hello", "model": "llama3.2:3b", "temperature": 0.5}
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)

def test_chat_endpoint():
    payload = {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "llama3.2:3b"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert "response" in response.json()