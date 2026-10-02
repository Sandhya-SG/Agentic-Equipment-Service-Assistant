from app.services.llm_client import generate_reply


def test_generate_reply_ollama_noop_when_not_configured(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("MODEL_NAME", "llama3.1")

    class DummyResponse:
        def __init__(self):
            self.status = 200

        def read(self):
            return b'{"response":"ok from model"}'

    def fake_urlopen(req, timeout=30):
        return DummyResponse()

    import app.services.llm_client as llm_client

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)

    result = generate_reply("hello")

    assert "ok from model" in result


def test_generate_reply_handles_empty_message():
    result = generate_reply("   ")
    assert result == ""
