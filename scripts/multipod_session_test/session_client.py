"""Sends conversations to the backend Service, a new connection per message, and reports where each one escalated.

Like the browser, it keeps the signed `clarification_state` a clarification response returns and sends it back with
the next message (CARRY_STATE=1, the default). CARRY_STATE=0 does not, which reproduces the original problem: the
state then lives only in the memory of the pod that asked.
"""

import collections
import json
import os
import urllib.request
import uuid

URL = os.getenv("BACKEND", "http://backend:8000") + "/api/chat"
N = int(os.getenv("CONVERSATIONS", "20"))
CARRY_STATE = os.getenv("CARRY_STATE", "1") == "1"
MSGS = 4


def post(cid, text, state):
    body = {"message": text, "equipment_model": "thermal_station", "conversation_id": cid}
    if state:
        body["clarification_state"] = state
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    return data["status"], data.get("clarification_state")


escalated_at = collections.Counter()
for n in range(N):
    cid = f"mp-{uuid.uuid4().hex[:10]}"
    statuses, state = [], None
    for i in range(MSGS):
        status, signed = post(cid, "It is not working" if i == 0 else f"The heater, detail {i}", state)
        statuses.append(status)
        if CARRY_STATE:
            if status == "clarification":
                state = signed
            elif status in ("ok", "halted", "escalated"):
                state = None
    first = next((i + 1 for i, s in enumerate(statuses) if s == "escalated"), None)
    escalated_at[first] += 1
    print(n, statuses)
print("SUMMARY escalated at message:", dict(sorted(escalated_at.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))))
print("same-pod behaviour (expected): escalates at message 3 in every conversation")
