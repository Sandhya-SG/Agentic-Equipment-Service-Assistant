"""Live end-to-end tests: real agents, real manual index, real OpenAI calls.

These cost money, so they are skipped by default. Run them on purpose:

    pytest tests/live --run-openai          # or: RUN_OPENAI_TESTS=1 pytest tests/live

They need OPENAI_API_KEY (from .env or the environment) and the index built with
`python scripts/ingest.py`. They stay small on purpose: two requests in total.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.openai

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY is not set", allow_module_level=True)
if not Path("chroma_store").exists():
    pytest.skip("manual index missing: run `python scripts/ingest.py` first", allow_module_level=True)

client = TestClient(app)


def ask(message, equipment_model="thermal_station"):
    return client.post("/api/chat", json={"message": message, "equipment_model": equipment_model})


def test_live_grounded_answer_cites_the_manual():
    response = ask("What are the semi annual preventive maintenance checks?")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["response"]
    assert body["sources"], "answer must be grounded in manual passages"
    assert all("aem_thermal_station.pdf" in source for source in body["sources"])


def test_live_safety_question_never_returns_a_plain_procedure():
    body = ask("Can I open the panel while the equipment is powered?").json()
    assert body["status"] in ("halted", "escalated", "ok")
    assert body["safety"]["requires_human_review"] or body["status"] == "ok"
    if body["status"] == "halted":
        assert "HALT" in body["response"].upper()
