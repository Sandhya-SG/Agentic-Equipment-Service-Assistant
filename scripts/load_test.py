"""Light load test for the backend (standard library only).

Two modes:
  --mode health  GET /health. No model call, no cost. Shows that the Service spreads load and drives the autoscaler.
  --mode chat    POST /api/chat with real questions. SPENDS OPENAI CREDIT, so --max-requests is a hard cap.

Run it against the backend Service through a port-forward:
  kubectl -n aem port-forward svc/backend 8000:8000
  python scripts/load_test.py --mode health --users 20 --duration 60
  python scripts/load_test.py --mode chat --users 5 --duration 120 --max-requests 60

It prints requests per second, median and 95th-percentile latency and the error rate, then asks the
backend to re-walk its audit chains (/api/monitoring?verify=true). Each pod writes its own chain, so
the check reaches whichever pod answers; run it a few times after a test, or read the pod logs.
"""

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request

QUESTIONS = [
    "What is the maintenance interval for the hydraulic filter?",
    "The unit shows an overheating warning. What should I check first?",
    "What PPE is needed when servicing the compressor?",
    "How do I reset the fault code after replacing the sensor?",
    "What is the recommended lubricant for the main bearing?",
]


def call(url: str, mode: str, n: int, timeout: float) -> tuple[bool, float]:
    start = time.perf_counter()
    try:
        if mode == "health":
            req = urllib.request.Request(f"{url}/health")
        else:
            body = json.dumps({"message": QUESTIONS[n % len(QUESTIONS)], "conversation_id": f"load-{n}"}).encode()
            req = urllib.request.Request(f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            ok = resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        ok = False
    return ok, time.perf_counter() - start


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--mode", choices=["health", "chat"], default="health")
    p.add_argument("--users", type=int, default=10, help="concurrent users")
    p.add_argument("--duration", type=int, default=60, help="seconds")
    p.add_argument("--max-requests", type=int, default=0, help="hard cap on requests (0 = none); use it for chat")
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()

    if args.mode == "chat" and not args.max_requests:
        raise SystemExit("chat mode spends OpenAI credit: set --max-requests")

    lock = threading.Lock()
    results: list[tuple[bool, float]] = []
    counter = [0]
    stop_at = time.perf_counter() + args.duration

    def worker() -> None:
        while time.perf_counter() < stop_at:
            with lock:
                if args.max_requests and counter[0] >= args.max_requests:
                    return
                n = counter[0]
                counter[0] += 1
            r = call(args.url, args.mode, n, args.timeout)
            with lock:
                results.append(r)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - started

    lat = sorted(t for ok, t in results if ok)
    errors = sum(1 for ok, _ in results if not ok)
    print(f"mode={args.mode} users={args.users} elapsed={elapsed:.1f}s requests={len(results)} errors={errors}")
    if results:
        print(f"requests/s : {len(results) / elapsed:.2f}")
        print(f"error rate : {errors / len(results):.1%}")
    if lat:
        print(f"median     : {statistics.median(lat):.3f}s")
        print(f"p95        : {lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.3f}s")

    try:
        with urllib.request.urlopen(f"{args.url}/api/monitoring?verify=true&limit=1", timeout=30) as resp:
            audit = json.load(resp).get("audit_integrity")
        print(f"audit chain: {audit}")
    except Exception as exc:  # the check is informative; do not fail the test run because of it
        print(f"audit chain: could not be checked ({exc})")


if __name__ == "__main__":
    main()
