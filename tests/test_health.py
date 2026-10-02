from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health_route():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_route():
    response = client.post(
        "/api/chat",
        json={"message": "hello"},
    )
    assert response.status_code == 200
    assert "response" in response.json()
