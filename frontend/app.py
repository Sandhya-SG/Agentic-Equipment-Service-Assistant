import json
import os
import re
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


# --- Human review: the page, and a proxy to the backend's review API ---------------------------------------------
# The proxy forwards the reviewer's headers and nothing more: the token is validated by the BACKEND, never here.
REVIEW_HEADERS = ("X-Reviewer-Token", "X-Reviewer-Id")
REVIEW_ID = re.compile(r"^[0-9a-f]{32}$")


@app.route("/review")
def review_page():
    return render_template("review.html")


def _backend_json(method, path, body=None, headers=None, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib_request.Request(
        f"{BACKEND_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            return jsonify(json.loads(response.read().decode("utf-8"))), response.status
    except urllib_error.HTTPError as exc:
        try:
            return jsonify(json.loads(exc.read().decode("utf-8"))), exc.code
        except ValueError:
            return jsonify({"detail": f"Backend returned HTTP {exc.code}"}), 502
    except Exception as exc:  # pragma: no cover - network failure path
        return jsonify({"detail": str(exc)}), 502


def _reviewer_headers():
    return {name: request.headers[name] for name in REVIEW_HEADERS if request.headers.get(name)}


@app.route("/api/review/items")
def review_items():
    status = request.args.get("status", "pending")
    if status not in ("pending", "decided", "all"):
        return jsonify({"detail": "status must be pending, decided or all"}), 400
    return _backend_json("GET", f"/api/reviews?status={status}", headers=_reviewer_headers())


@app.route("/api/review/items/<review_id>")
def review_item(review_id):
    if not REVIEW_ID.fullmatch(review_id):
        return jsonify({"detail": "no such review"}), 404
    return _backend_json("GET", f"/api/reviews/{review_id}", headers=_reviewer_headers())


@app.route("/api/review/items/<review_id>/decision", methods=["POST"])
def review_decision(review_id):
    if not REVIEW_ID.fullmatch(review_id):
        return jsonify({"detail": "no such review"}), 404
    body = request.get_json(silent=True) or {}
    return _backend_json("POST", f"/api/reviews/{review_id}/decision", body=body, headers=_reviewer_headers())


@app.route("/api/review/status/<review_id>")
def review_status(review_id):
    """For the engineer's banner: no reviewer headers, the backend shows only the state and the outcome."""
    if not REVIEW_ID.fullmatch(review_id):
        return jsonify({"detail": "no such review"}), 404
    return _backend_json("GET", f"/api/review-status/{review_id}")


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
