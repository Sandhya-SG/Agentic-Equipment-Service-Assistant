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


def test_generate_reply_uses_configured_base_url(monkeypatch):
    import app.services.llm_client as llm_client

    seen = {}

    class DummyResponse:
        def read(self):
            return b'{"response":"hi"}'

    def fake_urlopen(req, timeout=30):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(llm_client.settings, "ollama_base_url", "http://ollama:11434")
    monkeypatch.setattr(llm_client.settings, "ollama_timeout", 5.0)
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)

    assert generate_reply("hello") == "hi"
    assert seen == {"url": "http://ollama:11434/api/generate", "timeout": 5.0}


def test_generate_reply_retries_then_succeeds(monkeypatch):
    import urllib.error

    import app.services.llm_client as llm_client

    calls = {"n": 0}

    class DummyResponse:
        def read(self):
            return b'{"response":"recovered"}'

    def flaky_urlopen(req, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("down")
        return DummyResponse()

    monkeypatch.setattr(llm_client.settings, "ollama_max_retries", 2)
    monkeypatch.setattr(llm_client.settings, "ollama_retry_backoff", 0)
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", flaky_urlopen)

    assert generate_reply("hello") == "recovered"
    assert calls["n"] == 3


def test_generate_reply_falls_back_when_ollama_down(monkeypatch):
    import urllib.error

    import app.services.llm_client as llm_client

    def down_urlopen(req, timeout=30):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_client.settings, "ollama_max_retries", 1)
    monkeypatch.setattr(llm_client.settings, "ollama_retry_backoff", 0)
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", down_urlopen)

    assert generate_reply("hello") == llm_client.FALLBACK_MESSAGE


def test_generate_with_usage_reads_token_counts(monkeypatch):
    import app.services.llm_client as llm_client

    class DummyResponse:
        def read(self):
            return b'{"response":"hi","prompt_eval_count":12,"eval_count":5}'

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", lambda req, timeout=30: DummyResponse())

    result = llm_client.generate_with_usage("hello")

    assert result.text == "hi"
    assert (result.prompt_tokens, result.completion_tokens) == (12, 5)


def test_generate_with_usage_raises_when_model_unavailable(monkeypatch):
    import urllib.error

    import pytest

    import app.services.llm_client as llm_client
    from asa.graph.state import ModelUnavailable

    def down(req, timeout=30):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(llm_client.settings, "ollama_max_retries", 0)
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", down)

    with pytest.raises(ModelUnavailable):
        llm_client.generate_with_usage("hello")
