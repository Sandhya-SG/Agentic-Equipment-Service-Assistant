"""Explainability layer: citations with exact quotes, an evidence confidence and a decision explanation.

STARTER VERSION (8 Oct 2026): a first working implementation of the specification in
docs/explainability-spec.md, to be reviewed, tuned with subject-matter experts and extended by the
Explainability owner. The weights and thresholds below are provisional.

Design rules (from the spec):
  * a PURE function: no model call, no logging, no network, no file access, no change to `state`;
    so it adds no tokens and no latency and gives the same result for the same input;
  * it only restates evidence the agents already retrieved and never changes the answer, the status
    or the safety verdict;
  * every quote is an exact sentence of a retrieved manual passage (after collapsing white space);
    a citation that cannot be verified is not invented, it is counted as unverified;
  * the confidence is an *evidence confidence*, a transparent heuristic and not a probability.

Contract used by app/services/chat_service.py:
    explain(state, status) -> dict | None
returns None when there is nothing to explain (clarification, blocked, unavailable) and otherwise a
dict with at least `confidence` (0 to 1) and `confidence_level` ("low", "medium" or "high").
"""

from __future__ import annotations

import re
from typing import Any

from asa.graph.state import CONFIDENCE_FLOOR

CONFIDENCE_LOW = 0.50  # below this the level is "low"; from CONFIDENCE_FLOOR to 1.0 it is "high"

# Provisional weights (to be tuned with subject-matter experts).
BASE_CONFIDENCE = 0.50
BONUS_SUFFICIENT_EVIDENCE = 0.20
BONUS_SEVERAL_SECTIONS = 0.15
BONUS_ALL_CLAIMS_VERIFIED = 0.15  # scaled by the share of claims that were verified
PENALTY_PER_RETRY = 0.15
MAX_RETRY_PENALTY = 0.30
ESCALATED_CAP = 0.30
NO_CITATION_CAP = 0.40  # nothing could be quoted from the manual, so there is nothing to check
UNVERIFIED_CAP = 0.65  # a claim without a quoted manual sentence keeps the answer below CONFIDENCE_FLOOR ("verify")
CONTRADICTION_CAP = 0.45  # a claim that contradicts its manual sentence makes the answer "low"

MIN_SHARED_WORDS = 3  # a quote must share at least this many content words with the claim ...
MIN_SHORT_CLAIM_WORDS = 2  # ... or, for a shorter claim ("Electrical Hazards"), all of its words (never just one)
MIN_MATCH = 0.30  # ... and be at least this similar to it (cosine of the content-word sets)
MAX_SENTENCE_CHARS = 320  # longer "sentences" are merged table rows or lists, not quotable
MAX_ROW_LINES = 6  # a table row quote spans at most this many cells (the PDF puts one cell per line)
MAX_CITATIONS = 8
MAX_QUOTE_CHARS = 300
NO_EXPLANATION_STATUSES = {"clarification", "blocked", "unavailable"}

_SOURCE = re.compile(
    r"Source:\s*(?P<file>[^,\];]+?)\s*,\s*page\s*(?P<page>\d+)(?:\s*,\s*section\s*(?P<section>[^;\]]*))?",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "before after must should shall not do does can may if when any all each".split()
)
# Negation and numbers are what make a claim mean the opposite of, or something other than, its quote.
# The similarity match ignores negation words (they are in _STOP), so these are checked separately.
_NEGATION_WORDS = frozenset({"not", "no", "never", "cannot", "none", "nor", "without"})
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_BULLET = re.compile(r"^[\s\-\*•\d\.\)]+")
_HEADER = re.compile(
    r"^(safety decision|diagnostic assessment|documented|recommended|required)\b.*:\s*$", re.IGNORECASE
)


# ------------------------------------------------------------------ helpers
def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _norm(text: str) -> str:
    """Collapse all white space so a sentence that spans a line break still matches its passage."""
    return " ".join(str(text).split())


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(str(text).lower()) if w not in _STOP and len(w) > 1}


_PASSAGE_SPLIT = re.compile(r"(?<=[.!?])\s+|\s[•●\-]\s")


def _sentences(text: str) -> list[str]:
    """Sentences of a manual passage. PDF text wraps lines in the middle of sentences, so the white
    space is collapsed first and sentences are split on their punctuation (and on bullets)."""
    return [s.strip() for s in _PASSAGE_SPLIT.split(_norm(text)) if len(s.strip()) >= 12]


def _clean_claim(text: str) -> str:
    text = _SOURCE.sub(" ", text)
    text = re.sub(r"\[\s*[;,]?\s*\]", " ", text)  # brackets left empty after removing the markers
    text = text.replace("**", "")
    return _norm(_BULLET.sub("", text))


# PDF page furniture that gets glued to the front of a sentence: "11-3 Rev 0 AEM Thermal Station", "Table 2-2".
_LEADING_FURNITURE = re.compile(
    r"^(?:\d+-\d+\s+|Rev\s+\d+\s+|AEM\s+(?:1kW\s+)?Thermal\s+(?:Station|Retrofit\s+System)\s+|Table\s+\d+-\d+\s+|\d+(?:\.\d+)+\s+)+",
    re.IGNORECASE,
)
_SECTION_NUMBER = re.compile(r"\b\d+(?:\.\d+)+\b")


def _is_furniture(sentence: str) -> bool:
    """Table-of-contents lines, merged table rows and page headers are not quotable sentences."""
    if len(sentence) > MAX_SENTENCE_CHARS:
        return True
    if len(_SECTION_NUMBER.findall(sentence)) >= 3:  # a table of contents line
        return True
    tokens = sentence.split()
    return bool(tokens) and sum(t[0].isdigit() for t in tokens) / len(tokens) > 0.30


_TOC_ENTRY = re.compile(r"^\d+(?:\.\d+)*\s+\S.*\s\d+-\d+$")  # "11.1 Preventive Maintenance 11-2"
_ALPHA_WORD = re.compile(r"[A-Za-z]{3,}")


def _table_rows(text: str) -> list[str]:
    """Quote candidates for passages laid out as tables.

    The PDF text puts every table cell on its own line and tables have no sentence punctuation, so a
    whole table collapses into one over-long "sentence". A row is a short run of consecutive lines;
    joined with single spaces it is still an exact stretch of the (white-space collapsed) passage.
    """
    lines = [" ".join(line.split()) for line in str(text).splitlines()]
    lines = [line for line in lines if line]
    rows = []
    for start in range(len(lines)):
        for size in range(1, MAX_ROW_LINES + 1):
            window = lines[start : start + size]
            if len(window) < size or _TOC_ENTRY.match(window[-1]):
                break
            row = " ".join(window)
            if len(row) > MAX_SENTENCE_CHARS:
                break
            rows.append(row)
    return rows


def _is_row_furniture(row: str) -> bool:
    """Specification rows are legitimately full of numbers, so the digit-share rule of _is_furniture
    does not apply; table-of-contents fragments and rows without real words are still not quotable."""
    return len(_SECTION_NUMBER.findall(row)) >= 2 or len(_ALPHA_WORD.findall(row)) < 2


def _clean_quote(sentence: str) -> str:
    """Drop page furniture from the FRONT of the sentence; what is left is still a contiguous
    stretch of the passage, so the exact-substring rule keeps holding."""
    return _LEADING_FURNITURE.sub("", sentence).strip()


def _negated(text: str) -> bool:
    tokens = _TOKEN.findall(str(text).lower().replace("’", "'"))
    return any(token in _NEGATION_WORDS or token.endswith("n't") for token in tokens)


def _contradicts(claim: str, quote: str) -> bool:
    """True when the quote cannot support the claim although their words match: one of them is
    negated and the other is not, or the claim states a number the quote does not contain."""
    if _negated(claim) != _negated(quote):
        return True
    return not set(_NUMBER.findall(claim)) <= set(_NUMBER.findall(quote))


def _sentence_candidates(chunks: list):
    for chunk in chunks:
        for sentence in _sentences(_get(chunk, "text", "") or ""):
            if not _is_furniture(sentence):
                yield _clean_quote(sentence), chunk


def _row_candidates(chunks: list):
    for chunk in chunks:
        for row in _table_rows(_get(chunk, "text", "") or ""):
            if not _is_row_furniture(row):
                yield _clean_quote(row), chunk


def _find_quote(claim: str, chunks: list) -> tuple[tuple[str, Any] | None, bool]:
    """The sentence of the given passages most similar to the claim (cosine of content words).

    Prose sentences are tried first; table rows (see `_table_rows`) only when no sentence supports
    the claim, so answers taken from specification tables can be verified too.

    Returns `(quote and chunk, or None, contradicted)`. A similar sentence that contradicts the claim
    (see `_contradicts`) is never used as its quote; `contradicted` is True when only such sentences
    matched, so the claim stays unverified and the user can be told why.
    """
    claim_words = _words(claim)
    if not claim_words:
        return None, False
    found, contradicted = _best_candidate(claim, claim_words, _sentence_candidates(chunks))
    if found is None:
        found, row_contradicted = _best_candidate(claim, claim_words, _row_candidates(chunks), rows=True)
        contradicted = contradicted or row_contradicted
    return (found, False) if found is not None else (None, contradicted)


def _best_candidate(
    claim: str, claim_words: set[str], candidates, rows: bool = False
) -> tuple[tuple[str, Any] | None, bool]:
    """The candidate quote most similar to the claim, skipping those that contradict it.

    For table rows, a row that starts with the claim's leading term (its row label, for example
    "PDU: ...") ranks above every row that does not, so a quote neither starts in the previous row
    nor in a column header; the similarity decides within each group.
    """
    best: tuple[tuple[bool, float], str, Any] | None = None
    contradicted = False
    claim_label = _ALPHA_WORD.search(claim)
    # A short claim (a hazard or PPE label) must share all of its words; a one-word claim never matches.
    required = max(MIN_SHORT_CLAIM_WORDS, min(MIN_SHARED_WORDS, len(claim_words)))
    for quote, chunk in candidates:
        words = _words(quote)
        shared = claim_words & words
        if len(shared) < required or not words:
            continue
        score = len(shared) / ((len(claim_words) * len(words)) ** 0.5)
        if score < MIN_MATCH:
            continue
        if _contradicts(claim, quote):
            contradicted = True
            continue
        label = _ALPHA_WORD.search(quote) if rows and claim_label else None
        starts_with_label = bool(label and label.group(0).lower() == claim_label.group(0).lower())
        rank = (starts_with_label, score)
        if best is None or rank > best[0]:
            best = (rank, quote, chunk)
    if best is None:
        return None, contradicted
    quote, chunk = best[1], best[2]
    if quote not in _norm(_get(chunk, "text", "")):  # exact-substring rule: never invent a quote
        return None, False
    return (quote[:MAX_QUOTE_CHARS], chunk), False


def _citation(chunk: Any, quote: str) -> dict:
    """One citation. `doc_id`, `section_id`, `revision` and `quote_span` are the fields of
    `asa.graph.state.Citation` (the shared-state contract); `quote_span` is the same text as `quote`."""
    return {
        "source_file": _get(chunk, "source_file", ""),
        "page": _get(chunk, "page", 0),
        "section": _get(chunk, "section_title", "Unknown"),
        "quote": quote,
        "chunk_id": _get(chunk, "chunk_id", ""),
        "doc_id": _get(chunk, "doc_id", ""),
        "section_id": _get(chunk, "section_id", ""),
        "revision": _get(chunk, "revision", ""),
        "quote_span": quote,
    }


# ------------------------------------------------------------------ claims
def _marked_sentences(answer: str) -> list[str]:
    """Sentences of the answer, each together with the citation markers that belong to it.

    The RAG agent usually writes the marker AFTER the full stop (`Claim. [Source: ...]`), so a
    plain sentence split leaves the marker on its own. A segment that holds only markers is
    therefore attached to the sentence before it, which is the claim it cites.
    """
    sentences: list[str] = []
    for segment in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if not segment.strip():
            continue
        if sentences and _SOURCE.search(segment) and not _clean_claim(segment):
            sentences[-1] = f"{sentences[-1]} {segment}"
        else:
            sentences.append(segment)
    return sentences


def _claims_from_markers(answer: str, chunks: list) -> list[tuple[str, list, bool]]:
    """RAG answers cite inline: `[Source: file, page N, section X]` before or after each claim."""
    claims = []
    for sentence in _marked_sentences(answer):
        found = list(_SOURCE.finditer(sentence))
        if not found:
            continue
        pairs = {(m.group("file").strip().lower(), int(m.group("page"))) for m in found}
        preferred = [c for c in chunks if (str(_get(c, "source_file", "")).lower(), _int(_get(c, "page", 0))) in pairs]
        claim = _clean_claim(sentence)
        if claim:
            claims.append((claim, preferred, True))  # strict: the cited page must be a retrieved passage
    return claims


def _claims_from_structure(
    state: dict,
    chunks: list,
) -> list[tuple[str, list, bool]]:
    """
    Build claims from specialist outputs that explicitly
    identify their supporting retrieved chunks.

    Diagnostic uses RankedCause and Step.
    Safety uses SafetyClaim.
    """

    by_id = {
        _get(chunk, "chunk_id"): chunk
        for chunk in chunks
    }

    claims = []

    structured_items = (
        [
            (cause, "cause")
            for cause
            in state.get(
                "root_causes"
            ) or []
        ]
        + [
            (step, "action")
            for step
            in state.get(
                "troubleshooting_steps"
            ) or []
        ]
        + [
            (claim, "text")
            for claim
            in state.get(
                "safety_claims"
            ) or []
        ]
    )

    for item, field in structured_items:

        text = _get(
            item,
            field,
            "",
        )

        preferred = [
            by_id[chunk_id]
            for chunk_id
            in (
                _get(
                    item,
                    "supporting_chunk_ids",
                    [],
                )
                or []
            )
            if chunk_id in by_id
        ]

        if text:
            claims.append(
                (
                    _clean_claim(text),
                    preferred,
                    False,
                )
            )

    return claims


def _claims_from_lines(answer: str, chunks: list) -> list[tuple[str, list, bool]]:
    """Fallback (for example the Safety Agent): each line of the answer is a claim, matched to any passage."""
    claims = []
    for line in re.split(r"\n+", answer):
        line = line.strip()
        if not line or _HEADER.match(line) or line.lower().startswith("safety decision"):
            continue
        claim = _clean_claim(line)
        if len(claim) >= 12:
            claims.append((claim, chunks, False))
    return claims


def _collect_claims(state: dict, chunks: list) -> list[tuple[str, list, bool]]:
    answer = str(state.get("final_answer") or "")
    return (
        _claims_from_markers(answer, chunks)
        or _claims_from_structure(state, chunks)
        or _claims_from_lines(answer, chunks)
    )


# ------------------------------------------------------------------ confidence and explanation
def _confidence_level(confidence: float) -> str:
    if confidence < CONFIDENCE_LOW:
        return "low"
    if confidence < CONFIDENCE_FLOOR:
        return "medium"
    return "high"


def _why(state: dict, status: str, n_sources: int, verified: int = 0, checked: int = 0) -> str:
    """Why the answer has this status. For RAG answers the counts are the real ones, so the text
    never claims more support than the citations show."""
    specialist = state.get("current_step")
    if status == "escalated":
        return str(
            state.get("escalation_reason") or "The evidence or the safety assessment requires a human decision."
        )[:300]
    if status == "halted":
        return str(state.get("safety_reason") or "The manual prohibits the requested action.")[:300]
    if specialist == "diagnostic":
        return "Candidate conditions and checks were validated against the cited passages; the root cause is not confirmed."
    if specialist == "safety":
        return str(state.get("safety_reason") or "The manual supports this when its documented controls are followed.")[
            :300
        ]
    retrieved = len(state.get("retrieved_chunks") or [])
    if not checked:
        return f"Answered from {retrieved} retrieved manual passage(s); no individual claim could be checked."
    if not verified:
        return f"None of the {checked} claim(s) could be matched to the {retrieved} retrieved manual passage(s)."
    return (
        f"Answered from {n_sources} cited manual passage(s); "
        f"{verified} of {checked} claim(s) were matched to a manual sentence."
    )


def explain(state: dict, status: str) -> dict | None:
    """Explain one answer: where it comes from, how strong the evidence is, and why this status."""
    if status in NO_EXPLANATION_STATUSES:
        return None
    chunks = list(state.get("retrieved_chunks") or [])

    citations: list[dict] = []
    seen: set[tuple] = set()
    verified = 0
    contradicted = 0
    # An escalation withholds the answer: quoting the manual would only support what the user is not shown.
    claims = [] if status == "escalated" else _collect_claims(state, chunks)
    for claim, preferred, strict in claims:
        candidates = preferred if strict else (preferred or chunks)
        found, contradiction = _find_quote(claim, candidates) if candidates else (None, False)
        if not found:
            contradicted += contradiction
            continue
        quote, chunk = found
        verified += 1
        key = (_get(chunk, "source_file", ""), _get(chunk, "page", 0), quote)
        if key not in seen:
            seen.add(key)
            citations.append(_citation(chunk, quote))

    unverified = len(claims) - verified
    sections = {(c["source_file"], c["section"]) for c in citations if c["section"] != "Unknown"}
    retries = _int(state.get("retry_count"))

    confidence = BASE_CONFIDENCE
    notes: list[str] = []
    if state.get("sufficiency"):
        confidence += BONUS_SUFFICIENT_EVIDENCE
        notes.append("The retrieved evidence was judged sufficient.")
    else:
        notes.append("The retrieved evidence was not judged sufficient.")
    if len(sections) >= 2:
        confidence += BONUS_SEVERAL_SECTIONS
        notes.append(f"The answer is supported by {len(sections)} different manual sections.")
    elif citations:
        notes.append("The answer rests on a single manual section.")
    if claims:
        confidence += BONUS_ALL_CLAIMS_VERIFIED * (verified / len(claims))
        if unverified == 0:
            notes.append(f"All {len(claims)} claim(s) were matched to a sentence of the manual.")
        else:
            notes.append(
                f"{unverified} of {len(claims)} claim(s) could not be matched to a manual sentence: verify them."
            )
        if contradicted:
            notes.append(
                f"{contradicted} claim(s) appear to contradict the cited manual sentence "
                "(negation or numbers differ): verify them."
            )
    else:
        notes.append("No individual claims could be checked against the manual.")
    if not citations:
        confidence = min(confidence, NO_CITATION_CAP)
        notes.append("No manual sentence could be quoted for this answer.")
    if retries:
        confidence -= min(PENALTY_PER_RETRY * retries, MAX_RETRY_PENALTY)
        notes.append(f"{retries} extra retrieval attempt(s) were needed.")
    if unverified and citations:
        # Without any citation NO_CITATION_CAP already applies, with its own note.
        confidence = min(confidence, UNVERIFIED_CAP)
        notes.append(
            f"Not every claim is backed by a quoted manual sentence, so the confidence is capped below {CONFIDENCE_FLOOR:.2f}."
        )
    if contradicted:
        confidence = min(confidence, CONTRADICTION_CAP)
    if status == "escalated" or state.get("escalated") or state.get("safety_verdict") == "escalate":
        confidence = min(confidence, ESCALATED_CAP)
        notes.append("The run escalated to a human, so the confidence is capped.")

    confidence = round(max(0.0, min(1.0, confidence)), 2)
    level = _confidence_level(confidence)
    if level == "low":
        notes.append("Low confidence: verify with a qualified engineer.")

    return {
        "confidence": confidence,
        "confidence_level": level,
        "decision": {
            "status": status,
            "specialist": state.get("current_step"),
            "verdict": state.get("safety_verdict") if state.get("current_step") == "safety" else None,
            "why": _why(state, status, len({c["chunk_id"] for c in citations}), verified, len(claims)),
        },
        "citations": citations[:MAX_CITATIONS],
        "claims_checked": len(claims),
        "claims_verified": verified,
        "notes": notes,
    }
