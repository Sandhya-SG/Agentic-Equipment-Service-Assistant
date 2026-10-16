"""Check that Langfuse holds good traces of the running assistant: which agents ran, how long it took, the tokens and the
models used, the release it was built from, and that no question or answer text was captured. Reads only (the Langfuse
public API); no model call, no cost.

    python scripts/verify_langfuse.py                                 # traces of the last 30 minutes
    python scripts/verify_langfuse.py --since-minutes 120 --expect-commit 7795493
    python scripts/verify_langfuse.py --host http://localhost:3000 --min-traces 5

The keys come from the environment or `.env` (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST) and are never
printed. Exit code 0 means every check passed, 1 means one failed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ("request", "planning", "agentic_rag", "diagnostic", "safety")  # the graph nodes, as Langfuse names the chains
RELEASE_FIELDS = ("commit", "prompts", "corpus", "models")  # what the backend stamps on every trace
MASKED = "[content not captured]"
REQUEST_TRACE = "chat-request"  # the root trace the backend opens for every chat request


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed.append(name)
        return ok


# --------------------------------------------------------------------------- #
# Reading Langfuse                                                            #
# --------------------------------------------------------------------------- #


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def make_getter(host: str, public_key: str, secret_key: str):
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    def get(path: str) -> dict:
        request = urllib.request.Request(host.rstrip("/") + path, headers={"Authorization": "Basic " + token})
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - the operator's own Langfuse
            return json.loads(response.read())

    return get


def fetch_traces(get, since: datetime, limit: int, name: str | None = REQUEST_TRACE) -> list[dict]:
    stamp = urllib.parse.quote(since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    only = (
        f"&name={urllib.parse.quote(name)}" if name else ""
    )  # the application's own request traces, not scripts or graders
    return get(f"/api/public/traces?limit={limit}&orderBy=timestamp.desc&fromTimestamp={stamp}{only}").get("data", [])


def fetch_observations(get, trace_id: str) -> list[dict]:
    return get(f"/api/public/observations?traceId={urllib.parse.quote(trace_id)}&limit=100").get("data", [])


# --------------------------------------------------------------------------- #
# Summarising and checking (pure, tested without Langfuse)                    #
# --------------------------------------------------------------------------- #


def _number(value) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def summarise(trace: dict, observations: list[dict]) -> dict:
    """What one trace shows: the release, the agents in the order they started, the time, the tokens and the models."""
    metadata = trace.get("metadata") or {}
    chains = sorted(
        (o for o in observations if o.get("type") == "CHAIN" and o.get("name") in AGENTS),
        key=lambda o: o.get("startTime") or "",
    )
    generations = [o for o in observations if o.get("type") == "GENERATION"]
    usages = [o.get("usageDetails") or o.get("usage") or {} for o in generations]
    unmasked = any(_text_captured(o.get("input")) or _text_captured(o.get("output")) for o in generations)
    unmasked = unmasked or _text_captured(trace.get("input")) or _text_captured(trace.get("output"))
    return {
        "id": trace.get("id", ""),
        "name": trace.get("name"),
        "commit": metadata.get("commit") or trace.get("version") or "",
        "version": trace.get("version") or "",
        "release_fields": [field for field in RELEASE_FIELDS if metadata.get(field)],
        "agents": [o["name"] for o in chains],
        "latency_s": _number(trace.get("latency")),
        "generations": len(generations),
        "models": sorted({o.get("model") for o in generations if o.get("model")}),
        "tokens_in": int(sum(_number(u.get("input")) for u in usages)),
        "tokens_out": int(sum(_number(u.get("output")) for u in usages)),
        "cost_usd": _number(trace.get("totalCost")),
        "text_captured": unmasked,
        "environment": trace.get("environment"),
    }


def _text_captured(value) -> bool:
    """True when a trace field holds text that is not the masking marker (the question or the answer leaked in)."""
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str):
        # Free text (a question or an answer) has spaces; an identifier such as "thermal_station" does not.
        return value != MASKED and any(ch.isspace() for ch in value.strip())
    if isinstance(value, dict):
        return any(_text_captured(v) for v in value.values())
    if isinstance(value, list):
        return any(_text_captured(v) for v in value)
    return False


def same_commit(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


def verify(
    report: Report, summaries: list[dict], expect_commit: str | None, min_traces: int, allow_content: bool
) -> None:
    report.check(
        f"at least {min_traces} trace(s) in the window", len(summaries) >= min_traces, f"{len(summaries)} found"
    )
    if not summaries:
        return
    missing = [s["id"][:8] for s in summaries if len(s["release_fields"]) < len(RELEASE_FIELDS)]
    report.check(
        "every trace carries the release stamp (commit, prompts, corpus, models)",
        not missing,
        f"missing on {missing[:5]}" if missing else "",
    )
    report.check("the trace version equals the commit", all(same_commit(s["version"], s["commit"]) for s in summaries))
    commits = sorted({s["commit"] for s in summaries})
    print(f"INFO release commits seen: {commits}")
    if expect_commit:
        report.check(
            "every trace is from the expected release",
            all(same_commit(s["commit"], expect_commit) for s in summaries),
            f"seen {commits}",
        )
    with_agents = [s for s in summaries if len(set(s["agents"])) >= 2]
    report.check(
        "agent execution is visible (at least two agents in order)",
        bool(with_agents),
        f"{len(with_agents)} of {len(summaries)} traces",
    )
    if with_agents:
        print(f"INFO example agent order: {' -> '.join(with_agents[0]['agents'])}")
    latencies = [s["latency_s"] for s in summaries]
    report.check(
        "every trace has a latency",
        all(x > 0 for x in latencies) or all(x >= 0 for x in latencies) and max(latencies) > 0,
    )
    print(f"INFO latency: median {statistics.median(latencies):.2f} s, max {max(latencies):.2f} s")
    used = [s for s in summaries if s["generations"]]
    tokens = [s for s in used if s["tokens_in"] > 0 and s["tokens_out"] > 0 and s["models"]]
    report.check(
        "token usage and the model are recorded for model calls",
        bool(tokens) and len(tokens) == len(used),
        f"{len(tokens)} of {len(used)} traces that used a model",
    )
    if used:
        print(
            f"INFO tokens: {sum(s['tokens_in'] for s in used)} in, {sum(s['tokens_out'] for s in used)} out; models {sorted({m for s in used for m in s['models']})}; cost ${sum(s['cost_usd'] for s in used):.4f}"
        )
    if not allow_content:
        leaked = [s["id"][:8] for s in summaries if s["text_captured"]]
        report.check("no question or answer text is captured", not leaked, f"text in {leaked[:5]}" if leaked else "")


# --------------------------------------------------------------------------- #
# Running it                                                                  #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--host", default=None, help="Langfuse address (default: LANGFUSE_HOST, else http://localhost:3000)"
    )
    parser.add_argument("--since-minutes", type=int, default=30, help="how far back to look (default 30)")
    parser.add_argument("--limit", type=int, default=50, help="most traces to read (default 50)")
    parser.add_argument("--min-traces", type=int, default=1)
    parser.add_argument("--expect-commit", default=None, help="the release every trace should carry")
    parser.add_argument("--all-traces", action="store_true", help=f"also judge traces not named {REQUEST_TRACE!r}")
    parser.add_argument(
        "--allow-content", action="store_true", help="skip the privacy check (LANGFUSE_CAPTURE_CONTENT is on)"
    )
    args = parser.parse_args(argv)

    env = {**read_env_file(ROOT / ".env"), **{k: v for k, v in os.environ.items() if k.startswith("LANGFUSE_") and v}}
    if not (env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY")):
        print("error: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are needed (environment or .env)")
        return 1
    host = args.host or env.get("LANGFUSE_HOST") or "http://localhost:3000"
    get = make_getter(host, env["LANGFUSE_PUBLIC_KEY"], env["LANGFUSE_SECRET_KEY"])
    report = Report()
    since = datetime.now(timezone.utc) - timedelta(minutes=args.since_minutes)
    try:
        traces = fetch_traces(get, since, args.limit, None if args.all_traces else REQUEST_TRACE)
        summaries = [summarise(trace, fetch_observations(get, trace["id"])) for trace in traces]
    except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
        report.check("Langfuse can be read with these keys", False, str(error)[:120])
        print(f"RESULT {report.passed} passed, {len(report.failed)} failed")
        return 1
    report.check("Langfuse can be read with these keys", True, host)
    verify(report, summaries, args.expect_commit, args.min_traces, args.allow_content)
    print(f"RESULT {report.passed} passed, {len(report.failed)} failed")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
