import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# The agents search the manual and may retry, so a reply can take a while.
BACKEND_TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "120"))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Equipment Service Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px auto; max-width: 820px; padding: 0 16px; }
        select, textarea { width: 100%; margin-top: 10px; font: inherit; padding: 8px; box-sizing: border-box; }
        textarea { height: 110px; }
        button { margin-top: 10px; padding: 10px 20px; }
        button:disabled { opacity: .6; }
        #result { margin-top: 20px; }
        .card { padding: 15px; background: #f4f4f4; border-radius: 8px; border-left: 6px solid #888; }
        .card.ok { border-color: #2e7d32; }
        .card.clarification { border-color: #1565c0; }
        .card.halted, .card.escalated { border-color: #c62828; }
        .card.blocked, .card.unavailable, .card.error { border-color: #ef6c00; }
        .badge { display: inline-block; font-size: 12px; font-weight: bold; text-transform: uppercase;
                 padding: 2px 8px; border-radius: 10px; background: #ddd; margin-bottom: 8px; }
        .answer { white-space: pre-wrap; }
        .warn { margin-top: 12px; padding: 10px; background: #fff3e0; border-radius: 6px; }
        .meta { margin-top: 12px; font-size: 13px; color: #444; }
        .meta ul { margin: 4px 0 0 18px; padding: 0; }
    </style>
</head>
<body>
    <h1>Equipment Service Assistant</h1>

    <form id="chat-form">
        <label for="equipment">Equipment</label>
        <select id="equipment" required>
            <option value="">Loading equipment...</option>
        </select>
        <textarea id="message" placeholder="Describe the issue or ask a question about the manual..." required></textarea><br>
        <button type="submit" id="send">Send</button>
    </form>

    <div id="result"></div>

    <script>
        const resultBox = document.getElementById("result");
        const select = document.getElementById("equipment");
        const sendButton = document.getElementById("send");
        const messageBox = document.getElementById("message");
        const DEFAULT_PLACEHOLDER = messageBox.placeholder;
        // After a clarifying question the answer is sent as a new, complete question.
        function setClarifying(on) {
            messageBox.placeholder = on
                ? "Answer the question above with full detail (it is sent as a new question)..."
                : DEFAULT_PLACEHOLDER;
        }

        function el(tag, className, text) {
            const node = document.createElement(tag);
            if (className) node.className = className;
            if (text !== undefined) node.textContent = text;
            return node;
        }

        function addList(parent, title, items) {
            if (!items || !items.length) return;
            const box = el("div", "meta");
            box.appendChild(el("strong", "", title));
            const list = el("ul");
            items.forEach(function (item) { list.appendChild(el("li", "", item)); });
            box.appendChild(list);
            parent.appendChild(box);
        }

        // Model text is always inserted with textContent, never as HTML.
        function render(data) {
            resultBox.replaceChildren();
            const status = data.status || "error";
            const card = el("div", "card " + status);
            card.appendChild(el("span", "badge", status));
            card.appendChild(el("div", "answer", data.response || data.error || JSON.stringify(data, null, 2)));

            const safety = data.safety || {};
            if (safety.warning) card.appendChild(el("div", "warn", safety.warning));
            addList(card, "Required PPE", safety.ppe_required);
            addList(card, "Sources", data.sources);
            if (data.escalation_reason) card.appendChild(el("div", "meta", "Escalation reason: " + data.escalation_reason));
            if (data.run_id) card.appendChild(el("div", "meta", "Run ID: " + data.run_id));
            resultBox.appendChild(card);
        }

        async function loadEquipment() {
            try {
                const res = await fetch("/api/equipment");
                const items = await res.json();
                select.replaceChildren(new Option("Select equipment...", ""));
                items.forEach(function (item) { select.appendChild(new Option(item.label, item.id)); });
            } catch (error) {
                select.replaceChildren(new Option("Could not load equipment", ""));
            }
        }

        document.getElementById("chat-form").addEventListener("submit", async function (e) {
            e.preventDefault();
            sendButton.disabled = true;
            resultBox.replaceChildren(el("div", "card", "Searching the manual..."));

            try {
                const res = await fetch("/api/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        message: messageBox.value,
                        equipment_model: select.value || null
                    })
                });
                const data = await res.json();
                render(data);
                setClarifying(data.status === "clarification");
                if (data.status === "clarification") messageBox.value = "";
            } catch (error) {
                render({ status: "error", error: "Error: " + error.message });
            } finally {
                sendButton.disabled = false;
            }
        });

        loadEquipment();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


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
    body = {"message": payload.get("message", ""), "equipment_model": payload.get("equipment_model")}

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
