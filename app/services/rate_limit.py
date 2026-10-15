"""A per-client request limit for the chat endpoint: every chat request can cost model calls, so it is also a cost control.

A sliding window kept in this process's memory: at most RATE_LIMIT_PER_MINUTE requests per client in any 60 seconds
(default 30; 0 turns it off). With several backend pods each pod counts on its own, so the effective limit for a
client is up to the limit times the number of pods. That is enough to stop a runaway script or a spammed endpoint; a
shared limit would need a shared store (see the production notes in the runbook).

The client is the connecting address. Behind the frontend every request arrives from the frontend, so the frontend
passes the browser's address in X-Forwarded-For and the backend reads it only when TRUST_FORWARDED_FOR is "true"
(set it only where the backend is reachable through the frontend alone, as in the compose file and the cluster).
"""

import os
import threading
import time
from collections import deque

WINDOW_SECONDS = 60.0
MAX_TRACKED_CLIENTS = 10_000  # bounds memory when many different addresses connect

_lock = threading.Lock()
_hits: dict[str, deque] = {}


def limit_per_minute() -> int:
    try:
        return max(0, int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")))
    except ValueError:
        return 30


def client_key(peer: str | None, forwarded_for: str | None) -> str:
    """The address to count: the first X-Forwarded-For entry when forwarding is trusted, otherwise the peer."""
    if forwarded_for and os.getenv("TRUST_FORWARDED_FOR", "").lower() == "true":
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first[:64]
    return peer or "unknown"


def check(key: str, now: float | None = None) -> float:
    """Count a request. Returns 0 when it is allowed, otherwise the seconds to wait before the next one is."""
    limit = limit_per_minute()
    if limit == 0:
        return 0.0
    now = time.monotonic() if now is None else now
    cutoff = now - WINDOW_SECONDS
    with _lock:
        window = _hits.get(key)
        if window is None:
            if len(_hits) >= MAX_TRACKED_CLIENTS:
                _drop_idle(cutoff)
            window = _hits[key] = deque()
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= limit:
            return max(0.001, window[0] + WINDOW_SECONDS - now)
        window.append(now)
        return 0.0


def _drop_idle(cutoff: float) -> None:
    for key in [k for k, w in _hits.items() if not w or w[-1] <= cutoff]:
        del _hits[key]
    if len(_hits) >= MAX_TRACKED_CLIENTS:  # still full of active clients: forget the oldest rather than grow
        del _hits[next(iter(_hits))]


def reset() -> None:
    with _lock:
        _hits.clear()
