import json
import os
from urllib import request as urllib_request

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Equipment Service Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        textarea { width: 100%; height: 120px; margin-top: 10px; }
        button { margin-top: 10px; padding: 10px 20px; }
        #response { margin-top: 20px; padding: 15px; background: #f4f4f4; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>Equipment Service Assistant</h1>

    <form id="chat-form">
        <textarea id="message" placeholder="Type your request here..."></textarea><br>
        <button type="submit">Send</button>
    </form>

    <div id="response">Waiting for response...</div>

    <script>
        document.getElementById("chat-form").addEventListener("submit", async function (e) {
            e.preventDefault();

            const message = document.getElementById("message").value;
            const responseBox = document.getElementById("response");

            responseBox.textContent = "Sending request...";

            try {
                const res = await fetch("/api/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: message })
                });

                const data = await res.json();
                responseBox.textContent = data.response || JSON.stringify(data, null, 2);
            } catch (error) {
                responseBox.textContent = "Error: " + error.message;
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/ask", methods=["POST"])
def ask_backend():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")

    data = json.dumps({"message": message}).encode("utf-8")

    req = urllib_request.Request(
        f"{BACKEND_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8")
            return jsonify(json.loads(body)), response.status
    except Exception as exc:  # pragma: no cover - network failure path
        return jsonify(
            {
                "status": "error",
                "error": str(exc),
            }
        ), 502
