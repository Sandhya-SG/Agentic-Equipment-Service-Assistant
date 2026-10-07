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


def test_ready_is_503_until_the_agents_have_loaded(monkeypatch):
    import app.routes.health as health

    monkeypatch.setattr(health, "is_ready", lambda: False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not ready"


def test_ready_is_200_once_the_agents_have_loaded(monkeypatch):
    import app.routes.health as health

    monkeypatch.setattr(health, "is_ready", lambda: True)
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_health_stays_ok_even_when_not_ready(monkeypatch):
    import app.routes.health as health

    monkeypatch.setattr(health, "is_ready", lambda: False)
    assert client.get("/health").status_code == 200
