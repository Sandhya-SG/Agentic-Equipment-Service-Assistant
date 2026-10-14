import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# The agents search the manual and may retry, so a reply can take a while.
BACKEND_TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "120"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/equipment")
def equipment():
    try:
        with urllib_request.urlopen(f"{BACKEND_URL}/api/equipment", timeout=10) as response:
            return jsonify(json.loads(response.read().decode("utf-8")))
    except Exception:  # pragma: no cover - network failure path
        return jsonify([]), 502


@app.route("/api/ask", methods=["POST"])
def ask_backend():
    payload = request.get_json(silent=True) or {}

    body = {
        "message": payload.get("message", ""),
        "equipment_model": payload.get("equipment_model"),
        "conversation_id": payload.get("conversation_id"),
        "clarification_state": payload.get("clarification_state"),
    }

    req = urllib_request.Request(
        f"{BACKEND_URL}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=BACKEND_TIMEOUT) as response:
            return jsonify(json.loads(response.read().decode("utf-8"))), response.status
    except urllib_error.HTTPError as exc:
        # 400 (blocked), 422 (invalid input) and 503 (unavailable) carry a useful JSON body.
        try:
            return jsonify(json.loads(exc.read().decode("utf-8"))), exc.code
        except ValueError:
            return jsonify({"status": "error", "error": f"Backend returned HTTP {exc.code}"}), 502
    except Exception as exc:  # pragma: no cover - network failure path
        return jsonify({"status": "error", "error": str(exc)}), 502
