import json

import frontend.app as frontend


client = frontend.app.test_client()


class FakeResponse:
    def __init__(
        self,
        body,
        status=200,
    ):
        self._body = json.dumps(
            body
        ).encode("utf-8")

        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False


class FakeErrorBody:
    def __init__(
        self,
        body,
    ):
        self._body = json.dumps(
            body
        ).encode("utf-8")

    def read(self):
        return self._body

    def close(self):
        pass


def test_ask_forwards_conversation_id(
    monkeypatch,
):
    captured = {}

    def fake_urlopen(
        req,
        timeout=None,
    ):
        captured["url"] = req.full_url
        captured["body"] = json.loads(
            req.data.decode("utf-8")
        )

        return FakeResponse(
            {
                "response":
                    "What symptom are you observing?",

                "conversation_id":
                    "conversation-123",

                "status":
                    "clarification",

                "safety": {
                    "hazards": [],
                    "ppe_required": [],
                    "requires_human_review": False,
                    "warning": None,
                },

                "sources": [],
                "trace": [],
            }
        )

    monkeypatch.setattr(
        frontend.urllib_request,
        "urlopen",
        fake_urlopen,
    )

    response = client.post(
        "/api/ask",
        json={
            "message":
                "It's not working.",

            "equipment_model":
                "thermal_station",

            "conversation_id":
                "conversation-123",
        },
    )

    assert response.status_code == 200

    assert (
        captured["body"]["message"]
        == "It's not working."
    )

    assert (
        captured["body"]["equipment_model"]
        == "thermal_station"
    )

    assert (
        captured["body"]["conversation_id"]
        == "conversation-123"
    )

    body = response.get_json()

    assert (
        body["conversation_id"]
        == "conversation-123"
    )

    assert (
        body["status"]
        == "clarification"
    )


def test_ask_preserves_backend_error_response(
    monkeypatch,
):
    error_body = {
        "response":
            "The assistant is temporarily unavailable.",

        "conversation_id":
            "conversation-123",

        "status":
            "unavailable",

        "safety": {
            "hazards": [],
            "ppe_required": [],
            "requires_human_review": False,
            "warning": None,
        },

        "sources": [],
        "trace": [],
    }

    def fake_urlopen(
        req,
        timeout=None,
    ):
        raise frontend.urllib_error.HTTPError(
            url=req.full_url,
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=FakeErrorBody(
                error_body
            ),
        )

    monkeypatch.setattr(
        frontend.urllib_request,
        "urlopen",
        fake_urlopen,
    )

    response = client.post(
        "/api/ask",
        json={
            "message":
                "What maintenance is required?",

            "equipment_model":
                "thermal_station",

            "conversation_id":
                "conversation-123",
        },
    )

    assert response.status_code == 503

    body = response.get_json()

    assert body["status"] == "unavailable"

    assert (
        body["conversation_id"]
        == "conversation-123"
    )

    assert (
        body["response"]
        == "The assistant is temporarily unavailable."
    )