# Explainability Strategy

**Status:** Proposal for team review · **Date:** 12 Oct 2026
**Scope:** The Explanation Module of the Agentic Equipment Service Assistant (AEM)
**Related code:** `src/asa/components/explanation.py`, `src/asa/graph/state.py`, `app/services/chat_service.py`

## 1. Purpose

The shared state (`state.py`) already defines what the Explanation Module must produce:

- **State fields** under "Explanation Module writes": `confidence`, `citations` and `final_answer`.
- **Citation type**: `Citation(doc_id, section_id, revision, quote_span)`, where `quote_span` holds the exact supporting text.
- **Confidence threshold**: `CONFIDENCE_FLOOR = 0.70`. Below it, the system should warn the user or escalate.

The module should:

1. **Cite exact evidence.** Link each part of the answer to the manual passage that supports it and quote that passage, not just a file and page.
2. **Score confidence** from the evidence: retrieval strength, whether the evidence was sufficient, retries, and the Safety Agent's verdict. Below 0.70 it warns or escalates.
3. **Explain the decision.** Say why the answer was allowed, halted or escalated, which hazards and PPE applied, and what was searched.
4. **Present it clearly** to technicians and to subject-matter experts (SMEs), drawing on the audit trail.
5. **Add no new facts.** Restate evidence only, and check that every quote really appears in the retrieved text.

This document compares the current implementation with that contract and proposes how to make the explanations reliable. The approach follows the course material from the Graduate Certificate in Architecting AI Systems.

## 2. Gap analysis

| Contract item | Current state |
|---|---|
| `confidence`, `citations`, `final_answer` written by the Explanation Module | ❌ `explain()` runs **after** the graph, in `chat_service`, and returns a separate dict. These state fields stay at their defaults. |
| `Citation(doc_id, section_id, revision, quote_span)` | ⚠️ Citations are plain dicts with `source_file`, `page`, `section`, `quote` and `chunk_id`. They have no `revision` and are not `Citation` objects. |
| `CONFIDENCE_FLOOR = 0.70` triggers a warning or escalation | ❌ The floor only labels an answer "high". `logging_sub.py` keeps its own copy of 0.70, so the value is defined twice. Nothing warns or escalates. |
| Explain what was searched | ❌ The RAG agent returns `final_query` and `retrieval_attempts`, but the graph node discards them. |
| Explain why this specialist was chosen | ⚠️ The planner writes `planning_reason` into state, but no explanation uses it. |
| Retrieval strength in the score | ❌ Chunk scores are not used, and `context_relevance` is never set. |
| Quotes really appear in the manual | ✅ An exact-substring check already exists. |
| Claims are actually supported by their quotes | ⚠️ Claims are matched to quotes by word overlap: at least 3 shared content words and a cosine similarity of at least 0.30. This **cannot detect negation**. "Do not open the panel while powered" and "open the panel while powered" share almost every word, which matters in a safety application. |

## 3. Strategy

### Principle 1: Explanations come from evidence, never from LLM self-narration

The IMDA *Model AI Governance Framework for Agentic AI* (v1.5) warns that *"chain-of-thought reasoning, which is sometimes used for explainability, is not analogous to human reasoning and may not be a faithful explanation of the agent's actions."*

So `explain()` stays as it is now: **deterministic, with no LLM call, and all text generated from templates over structured fields.** An LLM is used only offline, to *evaluate* explanations (Principle 7), never to write them. This also keeps the module free of extra tokens and latency, and its output repeatable.

### Principle 2: Run inside the graph, but explain what the user is finally shown

The ArchAAS Day 3 quality-attribute mapping lists explainability mechanisms as *"tracing, evidence and provenance store, audit logs, explanation service"*. The Day 4 logical architecture places a safety gate after the response.

- Add an **`explain` node** after each specialist and before `END`. It writes `confidence` and `citations` into state. This honours the contract and makes both visible in the Langfuse trace and the audit log.
- Keep a thin step in `chat_service` that writes the **decision explanation** (the "why") *after* the rule-based hazard check, because that check can still change the status. The current code makes this choice for the same reason, and it should be kept.

### Principle 3: Build an evidence record with verifiable provenance

- Fill in `Citation` properly by adding `chunk_id`, `source_file`, `page`, and the quote's start and end character positions in the chunk. The positions let the UI highlight the quote and let a test check it mechanically.
  - **Team decision needed:** the `state.py` header says the contract changes only by team agreement.
- Make "add no new facts" a **tested invariant**. A citation is kept only if its quote appears in its chunk's text (after collapsing whitespace) at the stored positions; otherwise it is dropped.
- A citation that points to a page that was not retrieved counts as **unverified**. This rule already exists and should be kept.

### Principle 4: Claim-to-evidence alignment that is safe for a safety application

Three levels, strongest first:

1. **Structured claims.** Diagnostic and Safety outputs already carry `supporting_chunk_ids` that an LLM has checked as SUPPORTED. These are the strongest evidence.
2. **RAG answers.** These must have both the inline `[Source: …]` marker and the word-overlap match.
3. **Two new deterministic checks:**
   - **Numbers and units.** Torque values, voltages and intervals in a claim must appear in its quote.
   - **Negation.** If the quote has "do not", "never" or "must not" next to the matched words, the claim must have it too.

A claim that fails either check is **unverified**, and the UI flags it.

### Principle 5: A confidence score that can explain itself and acts at the floor

The IMDA *Model AI Governance Framework* (2nd edition) says AI systems *"can be designed to report on the confidence level of their predictions, and explainability features could focus on why the AI model had a certain level of confidence."*

- **Return the score's parts, not just the total.** Report each factor and its contribution:
  - retrieval strength (the top reciprocal-rank-fusion score, and its margin over the next result);
  - the sufficiency judgement;
  - the share of claims verified;
  - how many manual sections the evidence spans;
  - the retry penalty;
  - caps from the safety verdict and from hazards.

  The UI can then say *why* the confidence is 0.62.
- **Use one constant.** Import `CONFIDENCE_FLOOR` from `state.py` everywhere and delete the copy in `logging_sub.py`.
- **Act on the floor, and only ever add caution.** This is the same principle as the existing hazard rules.

| Confidence | Action |
|---|---|
| ≥ 0.70 | Normal answer |
| 0.50 – 0.70 | Answer shown with a "Verify with a qualified engineer" banner, and `requires_human_review = true` |
| < 0.50, **or** the safety or diagnostic specialist scores below 0.70 | Escalate; actionable steps withheld |

- **Keep calling it "evidence confidence", not a probability,** until it has been calibrated (Principle 7).

### Principle 6: Design the explanations from user questions

XRAI Day 2 uses a question-driven design process (the *XAI Question Bank*). Its question types map onto this application as follows:

| XAI question | What the application shows | Data source |
|---|---|---|
| **Why** this answer? | The specialist, the planner's reason, and the verified citations | `planning_reason`, `citations` |
| **Why not** (why halted or escalated)? | Verdict, hazards, PPE, escalation reason, and hazards found by the rules | `safety_verdict`, `hazards`, `ppe_required`, `escalation_reason` |
| **How confident**? | The score and each factor's contribution | The confidence breakdown |
| **What was searched**? | Search queries, number of attempts, and the senior agent's decision | The RAG result (these fields need to be kept in state) |
| **Output scope** | "Advisory only: the assistant never operates equipment" | A short fact sheet (the `Step` type has no execution capability) |

Two views are built from the same data:

- **Technician view:** a compact panel with the verdict badge, the confidence with a one-line reason, PPE and hazards, and expandable highlighted quotes.
- **SME and auditor view:**
  - a claims table (claim → quote → chunk → verified?);
  - the search attempts;
  - the agent path with timings;
  - the `run_id`, linking to the audit log and the Langfuse trace.

  This view could be served by a `GET /api/runs/{run_id}/explanation` endpoint built from the audit trail.

**Guard against automation bias.** Both the IMDA Agentic AI framework and OWASP's agentic risk ASI09 ("human-agent trust exploitation") warn about it. Unverified claims and low confidence should be shown *more* prominently than verified ones. The application should never write persuasive prose that makes a weak answer look strong.

### Principle 7: Test and evaluate the explanations, not just display them

This follows ArchAAS Day 2's preferred order of evaluators (code first, then LLM-as-judge, then human review) and the AIS07 CI/CD quality gates.

- **Code checks (unit tests, run in CI):**
  - every quote appears in its chunk;
  - the explanation never changes the answer or the status;
  - escalated answers carry no quotes;
  - the confidence floor triggers the actions in Principle 5;
  - adversarial cases come out unverified: negated claims, wrong numbers, and citations to pages that were not retrieved.
- **LLM-as-judge (offline, not per request):** run a faithfulness metric (DeepEval, as taught in AIS07 LLMSecOps) over an evaluation set in CI.
- **Human review:**
  - Build an SME-labelled evaluation set. Start with about 20 questions, as recommended in ArchAAS Day 2, and grow it with real failures.
  - Use it to **calibrate the confidence weights**, which are currently marked provisional. Answers scored ≥ 0.70 should be correct far more often than those below. Plot accuracy by confidence band to check this.
- **Monitoring:**
  - Send an explanation-coverage score (verified claims as a share of all claims) to Langfuse, alongside `confidence`.
  - Cross-check `/api/feedback` ratings against confidence.
  - Keep the existing `low_confidence_rate` on `/api/monitoring`.

### Principle 8: Governance documentation

Write `docs/explainability-spec.md`, which `explanation.py` already refers to but which does not exist yet. It should cover:

- the question mapping (Principle 6);
- the confidence factors and the floor policy (Principle 5);
- the invariants (Principles 3 and 4);
- the calibration method (Principle 7).

Map it to the IMDA *Model AI Governance Framework* (2nd edition), §3.25–3.28 (explainability, repeatability, traceability and auditability), and to the transparency guidance in the IMDA *Model AI Governance Framework for Agentic AI*. Together this forms the explainability section of the Practice Module report.

## 4. Implementation phases

| Phase | Work | Outcome |
|---|---|---|
| **1. Contract and gate** | Add an `explain` node that writes `confidence` and typed `citations` into state. Use one `CONFIDENCE_FLOOR` and apply the action policy. Keep `final_query` and the number of retrieval attempts in state. Use `planning_reason`. Add tests. | The `state.py` contract is met, and low-confidence answers trigger warnings or escalation. |
| **2. Trustworthiness** | Number and negation checks. Confidence factor breakdown. Technician and SME views. The run-explanation endpoint. | Explanations can be checked and fit their audience. |
| **3. Validation** | SME evaluation set and calibration. DeepEval faithfulness gate in CI. Langfuse coverage score. `docs/explainability-spec.md`. | Evidence that the explanations and the confidence score can be trusted. |

Phase 1 is worth doing on its own: it makes the contract real and puts the 0.70 threshold to work.

## 5. References (course material)

- **IMDA**, *Model AI Governance Framework*, 2nd edition: §3.25–3.28 on explainability, repeatability, traceability and auditability; the guidance on reporting confidence levels.
- **IMDA**, *Model AI Governance Framework for Agentic AI* v1.5: the warning on chain-of-thought faithfulness, explainable outputs for user approval, transparency to users, and automation bias.
- **XRAI Day 2** (Explainable & Responsible AI), *Explainability, Bias, Fairness and Vulnerability of AI Systems*, and the reading *Question-Driven Design Process for Explainable AI User Experiences* (the XAI Question Bank).
- **ArchAAS Day 2** (Architecting Agentic AI Solutions): the agent evaluator and multimodal guardrails patterns, the preferred order of evaluators, and starting with about 20 evaluation examples.
- **ArchAAS Day 3**: the mapping of quality attributes to mechanisms, including explainability.
- **ArchAAS Day 4**: logical architecture, guardrails, and the safety gates around each response.
- **AIC Day 3** (AI & Cybersecurity), *Agentic AI Risks and Mitigations*: OWASP Top 10 for Agentic Applications, ASI09 human-agent trust exploitation.
- **AIS07** (Deploying and Operating AI Solutions), *CI/CD for Agentic AI Solutions* (quality gates) and *LLMSecOps* (DeepEval, groundedness and faithfulness metrics, Langfuse).
