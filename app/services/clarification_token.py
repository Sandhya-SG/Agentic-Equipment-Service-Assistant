"""Signed clarification state that travels with the conversation.

When the assistant asks a clarification question it used to remember the original question in the memory of one
backend process. With several backend pods a follow-up could reach a pod that remembered nothing (measured: 5 of 20
conversations worked with two pods). The pending state is now also returned to the client as a signed token; the client
sends it back with its next message and ANY pod can continue the conversation from it.

The token is `base64url(payload).base64url(HMAC-SHA256(secret, base64url(payload)))`. The payload holds the original
question, the question asked, the equipment, the number of clarification rounds, the conversation id and an expiry
time. A token that is altered, expired, made with another secret, or sent in another conversation is ignored and the
backend falls back to its in-memory state, exactly as before.

It needs `CLARIFICATION_SECRET`, the same value on every pod. Without it no token is issued and nothing changes.
`CLARIFICATION_TTL_S` (default 900) is how long a token stays valid.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from app.services.clarification_store import PendingClarification

VERSION = 1
DEFAULT_TTL_S = 900
MAX_TOKEN_CHARS = 8000  # the request schema allows the same length


def _secret() -> bytes | None:
    value = os.getenv("CLARIFICATION_SECRET", "").strip()
    return value.encode("utf-8") if value else None


def _ttl() -> int:
    try:
        ttl = int(os.getenv("CLARIFICATION_TTL_S", DEFAULT_TTL_S))
    except (TypeError, ValueError):
        return DEFAULT_TTL_S
    return ttl if ttl > 0 else DEFAULT_TTL_S


def enabled() -> bool:
    return _secret() is not None


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(secret: bytes, body: str) -> str:
    return _b64(hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest())


def issue(conversation_id: str, pending: PendingClarification, now: float | None = None) -> str | None:
    """The signed token for this pending clarification, or None when signing is not configured."""
    secret = _secret()
    if secret is None:
        return None
    payload = {
        "v": VERSION,
        "cid": conversation_id,
        "q": pending.original_query,
        "ask": pending.clarification_question,
        "eq": pending.equipment_model,
        "n": pending.clarification_count,
        "exp": int((time.time() if now is None else now) + _ttl()),
    }
    body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    token = f"{body}.{_sign(secret, body)}"
    return token if len(token) <= MAX_TOKEN_CHARS else None


def verify(token: str | None, conversation_id: str, now: float | None = None) -> PendingClarification | None:
    """The pending clarification a valid token carries, or None for anything else (never raises)."""
    secret = _secret()
    if secret is None or not isinstance(token, str) or not token or len(token) > MAX_TOKEN_CHARS:
        return None
    try:
        body, signature = token.split(".")
        if not hmac.compare_digest(signature, _sign(secret, body)):
            return None
        payload = json.loads(_unb64(body))
        if payload["v"] != VERSION or payload["cid"] != conversation_id:
            return None
        if payload["exp"] < (time.time() if now is None else now):
            return None
        original, ask, equipment, count = payload["q"], payload["ask"], payload["eq"], payload["n"]
        if (
            not isinstance(original, str)
            or not isinstance(ask, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
        ):
            return None
        if equipment is not None and not isinstance(equipment, str):
            return None
        return PendingClarification(
            original_query=original,
            clarification_question=ask,
            equipment_model=equipment,
            clarification_count=count,
        )
    except (ValueError, KeyError, TypeError, UnicodeError):
        return None
