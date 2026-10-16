import json
import os
import re
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

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


# --- Monitoring: the page, and a proxy to the backend's /api/monitoring ----------------------------------------------
# The proxy forwards the operator's token and nothing more: the BACKEND checks it. The page is only a readable view.


def _safe_link(value):
    """A link for the page only when it is an http(s) address (never javascript: or data:)."""
    parsed = urlparse((value or "").strip())
    return value.strip() if parsed.scheme in ("http", "https") and parsed.netloc else ""


@app.route("/monitor")
def monitor_page():
    return render_template("monitor.html", langfuse_url=_safe_link(os.getenv("LANGFUSE_URL", "")))


@app.route("/api/monitor/data")
def monitor_data():
    limit = request.args.get("limit", "200")
    verify = request.args.get("verify", "true")
    if not (limit.isdigit() and 1 <= int(limit) <= 5000) or verify not in ("true", "false"):
        return jsonify({"detail": "limit must be 1 to 5000 and verify true or false"}), 400
    token = request.headers.get("X-Monitor-Token")
    headers = {"X-Monitor-Token": token} if token else {}
    return _backend_json("GET", f"/api/monitoring?limit={int(limit)}&verify={verify}", headers=headers, timeout=60)


# --- Feedback: a helpful / not helpful rating of an answer, forwarded to the backend ------------------------------
# The browser sends only what the backend accepts (a 12-character run id, a boolean, an optional short comment); anything
# else is turned away here. Each backend pod keeps its own audit log and only a pod that has the run accepts its
# rating, so an "unknown run" answer is tried again (the Service may pick another pod) before it is reported.
RUN_ID = re.compile(r"^[0-9a-f]{12}$")
FEEDBACK_ATTEMPTS = 4
MAX_COMMENT_CHARS = 500


@app.route("/api/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True)
    payload = payload if isinstance(payload, dict) else {}
    run_id, helpful, comment = payload.get("run_id"), payload.get("helpful"), payload.get("comment")
    valid = (
        isinstance(run_id, str)
        and RUN_ID.fullmatch(run_id)
        and isinstance(helpful, bool)
        and (comment is None or (isinstance(comment, str) and len(comment) <= MAX_COMMENT_CHARS))
    )
    if not valid:
        return jsonify({"detail": "run_id (12 hex characters) and helpful (true or false) are required"}), 400
    body = {"run_id": run_id, "helpful": helpful}
    if comment:
        body["comment"] = comment
    for _ in range(FEEDBACK_ATTEMPTS):
        response, status = _backend_json("POST", "/api/feedback", body, timeout=30)
        if status != 404:
            break
    return response, status


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
        # The browser's address, so the backend can limit each client rather than the frontend as a whole.
        headers={"Content-Type": "application/json", "X-Forwarded-For": request.remote_addr or ""},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=BACKEND_TIMEOUT) as response:
            return jsonify(json.loads(response.read().decode("utf-8"))), response.status
    except urllib_error.HTTPError as exc:
        # 400 (blocked), 422 (invalid input), 429 (too many requests) and 503 (unavailable) carry a useful JSON body.
        try:
            return jsonify(json.loads(exc.read().decode("utf-8"))), exc.code
        except ValueError:
            return jsonify({"status": "error", "error": f"Backend returned HTTP {exc.code}"}), 502
    except Exception as exc:  # pragma: no cover - network failure path
        return jsonify({"status": "error", "error": str(exc)}), 502
