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


def test_chat_route(monkeypatch):
    import app.services.chat_service as chat_service

    monkeypatch.setattr(
        chat_service,
        "run_graph",
        lambda message, equipment_model: {"request_status": "READY", "final_answer": "hi there"},
    )
    response = client.post(
        "/api/chat",
        json={"message": "hello", "equipment_model": "thermal_station"},
    )
    assert response.status_code == 200
    assert "response" in response.json()
