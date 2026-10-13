# Promptfoo tests for the AEM assistant

Black-box tests against a running backend (`POST /api/chat` and friends). They check what a client sees: status codes, the response contract, blocking, safety outcomes, citations, explanation and latency.

## Suites

| File | What it checks | Key needed | Where it runs |
|---|---|---|---|
| `free-api.yaml` | Input validation (422), error contract, health, **every catalogued prompt-injection attack is blocked**, ordinary questions are not blocked | No (fake key is fine) | Every PR (`.github/workflows/promptfoo-tests-ci.yml`), gating |
| `known-gaps.yaml` | Documented weaknesses of the pattern guard: tricky legitimate questions that get blocked, attacks the guard cannot see | No | CI, **reported only, expected to fail** |
| `live-api.yaml` | Request decision and over-asking, routing, retrieval and equipment isolation, citations, explanation, grounding, safety on the seeded unsafe cases, injection resistance past the guard, disclosure | Real `OPENAI_API_KEY` | Locally, on demand (CI has no manuals) |
| `perf.yaml` | End-to-end latency per request type against the PRD target of about 10 s | Real key | Locally, on demand |

The datasets are shared with the offline evaluation (`eval/datasets/adversarial.jsonl`, `eval/datasets/safety_cases.jsonl`), so the API tests and `python eval/run_eval.py` always use the same attacks.

## How it works

- `provider.js` is a custom provider. It sends a **fresh `conversation_id` on every request** (the backend keeps a pending clarification per conversation. Until this was fixed in `app/main.py`, requests without an id shared one fixed id, and one caller's clarification leaked into the next caller's request; the fresh id keeps the tests independent regardless), records the HTTP status and latency, and can send raw or oversized bodies.
- `asserts.js` holds the assertions. Each returns a reason, so a failure says what was wrong.
- `generators.js` turns the datasets into tests.

## Run it

```bash
cd promptfoo-tests
npm ci                                  # installs the pinned promptfoo from package-lock.json

# 1. Start the backend (repository root). On Windows use PYTHONUTF8=1 and ";" in PYTHONPATH.
PYTHONUTF8=1 PYTHONPATH=.:src python -m uvicorn app.main:app --port 8000

# 2. Free suite (works with a fake OPENAI_API_KEY too)
npx promptfoo eval -c free-api.yaml --no-cache --no-table
npx promptfoo eval -c known-gaps.yaml --no-cache --no-table      # reported, expected to fail

# 3. Live suites (real key in OPENAI_API_KEY; the llm-rubric grader uses it as well)
npx promptfoo eval -c live-api.yaml --no-cache --no-table -j 3
npx promptfoo eval -c perf.yaml --no-cache --no-table -j 1 --repeat 3

npx promptfoo view                      # browse the last results
```

`BACKEND_URL` changes the target (default `http://localhost:8000`). Set `PROMPTFOO_DISABLE_TELEMETRY=1` to keep results local.

## Mapping to the evaluation types

| Evaluation | Where |
|---|---|
| Request | `live-api.yaml` (clear question proceeds, vague one is clarified, over-ask checks) |
| Routing | `live-api.yaml` (RAG, diagnostic, safety) |
| Retrieval | `live-api.yaml` (citations come from the selected manual only); retrieval hit rate is an offline evaluation |
| Grounding | `live-api.yaml` (no invented torque value, off-topic question) with an `llm-rubric` grader; the scored version is `deepeval-tests/` (faithfulness, relevancy, no invented specifics, 90% target) |
| Safety | `live-api.yaml`, one test per seeded unsafe case; rules themselves in the offline evaluation |
| Prompt injection | `free-api.yaml` (guard), `live-api.yaml` (resistance past the guard) |
| Explainability | `live-api.yaml` (confidence in 0 to 1, level matches the bands, trace order, inline citations) |
| Performance | `perf.yaml` |

## Findings these suites have reported

Details and evidence are in `docs/evaluation-test-cases-and-results.md`.

- **Fixed in this work:** requests without a `conversation_id` all shared the id `demo-session`, so unrelated API callers continued each other's pending clarification. Each request now gets its own id unless the client sends one (regression test in `tests/test_chat.py`).
- **Fixed on main by the team, confirmed by these suites:** the Request agent no longer over-asks on short clear questions (the two over-ask checks failed before and pass now), hazardous requests are no longer mostly met with a bare clarification (1 of 25, was 10), and every request type is under the 10 s target (the hazardous path went from 14 s to 7.8 s).
- **Still open:** the input guard blocks 6 of 24 legitimate questions and misses 10 obfuscated attacks (`known-gaps.yaml`); grounding is 87.5% against the 90% target, with one answer that says to reconnect the EGW quick connects where the manual says to disconnect them (DeepEval).
- **Windows only:** a request containing non-ASCII text returned 503 when the backend ran without `PYTHONUTF8=1`, because the agents print the text to a cp1252 console. Linux and Docker are not affected.

## Notes

- The Promptfoo version is pinned in `package.json` and `package-lock.json` (llmapp09 uses `@latest`). Update it on purpose, not by accident.
- Do not point the live suites at a shared or production backend: they send attack prompts and cost tokens.
- A test that always fails hides regressions, so the documented gaps live in `known-gaps.yaml` and not in the gating suite. A case that starts passing there can move to `free-api.yaml`.

## Routing stability

Routing is decided by a model, so the routing, retrieval and explanation tests were repeated five times each (`--repeat 5`): 40 of 40 passed. Re-run `npx promptfoo eval -c live-api.yaml --filter-pattern "^(routing|retrieval|explainability|request - a clear)" --repeat 5 --no-cache` after any change to the planning prompt.
