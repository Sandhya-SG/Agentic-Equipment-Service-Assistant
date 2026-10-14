"""Sends conversations to the backend Service, a new connection per message, and reports where each one escalated."""

import collections
import json
import os
import urllib.request
import uuid

URL = os.getenv("BACKEND", "http://backend:8000") + "/api/chat"
N = int(os.getenv("CONVERSATIONS", "20"))
MSGS = 4


def post(cid, text):
    req = urllib.request.Request(
        URL,
        data=json.dumps({"message": text, "equipment_model": "thermal_station", "conversation_id": cid}).encode(),
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["status"]


escalated_at = collections.Counter()
for n in range(N):
    cid = f"mp-{uuid.uuid4().hex[:10]}"
    statuses = [post(cid, "It is not working" if i == 0 else f"The heater, detail {i}") for i in range(MSGS)]
    first = next((i + 1 for i, s in enumerate(statuses) if s == "escalated"), None)
    escalated_at[first] += 1
    print(n, statuses)
print("SUMMARY escalated at message:", dict(sorted(escalated_at.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))))
print("same-pod behaviour (expected): escalates at message 3 in every conversation")
