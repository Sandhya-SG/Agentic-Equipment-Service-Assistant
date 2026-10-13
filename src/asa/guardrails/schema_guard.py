"""Response firewall: the last check on what the assistant says, before it leaves the system.

The input guard (injection.py) and the safety rules (safety_rules.py) look at what comes IN. Nothing looked at
what goes OUT. A model can still produce text that must not reach the engineer or the API caller: a credential,
personal data it was never given, a copy of its own instructions, or markup that a client might execute. This module
is that check (OWASP LLM02 sensitive information disclosure, LLM05 improper output handling, LLM07 system prompt
leakage, LLM10 unbounded output).

What it does to an answer, in order:

  1. secrets        keys, tokens, private keys, credential assignments: redacted.
  2. personal data  e-mail addresses, phone numbers, NRIC/FIN numbers, payment-card numbers: redacted.
  3. markup         script tags, iframes, javascript: links, inline event handlers: removed.
  4. prompt leak    the answer contains the system's own prompt text or an internal JSON structure: WITHHELD.
  5. length         an answer over the limit is cut and says so.

Evidence is data, not a licence. Personal-looking strings that appear verbatim in the manual passages the answer was
built from (the company's contact details on the cover page, a part number that looks like a phone number) are the
manual's own content and are kept. Provider-style keys and private keys are never kept, wherever they came from.

The result carries only categories ("email", "secret"), never the text that triggered them, so it can be written to
the audit trail and traces without leaking what it caught.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

MAX_ANSWER_CHARS = 8000

WITHHELD_MESSAGE = (
    "This answer could not be returned safely and was withheld. "
    "Please rephrase the question or contact a qualified engineer."
)
TRUNCATION_NOTICE = "\n\n[Answer shortened: it exceeded the length limit.]"

# --------------------------------------------------------------------------- #
# Secrets                                                                     #
# --------------------------------------------------------------------------- #

# Never kept, even if the same text appears in the evidence: a real key in a manual passage is a leak too.
_HARD_SECRETS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsk-(?:proj-|live-|test-|svcacct-)?[A-Za-z0-9_-]{20,}"), "[REDACTED KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "[REDACTED KEY]"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "[REDACTED KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), "[REDACTED KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "[REDACTED TOKEN]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*"), "Bearer [REDACTED TOKEN]"),
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"
        ),
        "[REDACTED PRIVATE KEY]",
    ),
]

# "password = hunter2": the manual's own default settings could look like this, so these are kept when the same
# text is in the evidence.
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\s*[:=]\s*[^\s,;]{3,}[^\s,;.)\]]"
)

# --------------------------------------------------------------------------- #
# Personal data                                                               #
# --------------------------------------------------------------------------- #

_EMAIL = re.compile(r"\b[\w.%+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")
# Phone numbers need a "+" country code, or the Singapore shape (3, 6, 8 or 9 first) written with a separator
# ("9123 4567"). A plain run of digits is never taken as a phone number, so part numbers, pressures and dates in
# equipment answers are left alone.
_PHONE = re.compile(
    r"(?<![\w.])(?:"
    r"\+65[\s-]?[3689]\d{3}[\s-]?\d{4}"  # +65 9123 4567
    r"|\+\d{1,3}[\s-]?\d{1,4}(?:[\s-]\d{2,4}){2,3}"  # +44 20 7946 0958
    r"|(?:65[\s-])?[3689]\d{3}[\s-]\d{4}"  # 9123 4567, 65 9123-4567
    r")(?!\w)(?!\.\d)"  # may end a sentence, but is not the start of a decimal
)
_NRIC = re.compile(r"\b[STFGM]\d{7}[A-Z]\b")
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def _luhn_valid(digits: str) -> bool:
    total, flip = 0, False
    for ch in reversed(digits):
        n = int(ch)
        if flip:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        flip = not flip
    return total % 10 == 0


# --------------------------------------------------------------------------- #
# Markup                                                                      #
# --------------------------------------------------------------------------- #

_MARKUP = [
    re.compile(r"(?is)<\s*script\b.*?(?:<\s*/\s*script\s*>|$)"),
    re.compile(r"(?is)<\s*(?:iframe|object|embed)\b[^>]*>(?:.*?<\s*/\s*(?:iframe|object|embed)\s*>)?"),
    re.compile(r"(?i)\bjavascript\s*:"),
    re.compile(r"(?i)\bon(?:error|load|click|mouseover|focus)\s*=\s*[\"']?[^\s>\"']+"),
]

# --------------------------------------------------------------------------- #
# The system's own text                                                       #
# --------------------------------------------------------------------------- #

# Distinctive phrases from the agents' prompts. An answer that contains one is repeating its instructions.
# tests/unit/test_schema_guard.py checks that every phrase is still present in the agent source, so this list cannot
# silently go stale when a prompt is reworded.
PROMPT_SIGNATURES = (
    "your only task is to determine whether the request",
    "decision rule (apply this first)",
    "choose exactly one specialist agent",
    "routing priority:",
    "you are the safety agent for an aem",
    "using only the supplied evidence, make a conservative",
    "return valid json only in this exact structure",
    "you are a conservative equipment",
    "you are the junior retrieval agent",
    "you are the senior retrieval agent",
    "answer the engineer using only the retrieved evidence",
    "the default answer is ready",
)
# The internal structures the agents exchange. Shown to an engineer they are a leak, not an answer.
_INTERNAL_JSON = re.compile(
    r'(?s)^\s*(?:\[\s*)?\{\s*"(?:verdict|candidate_causes|recommended_checks|tool_calls|supported)"\s*:'
)


@dataclass(frozen=True)
class OutputResult:
    """The checked answer. `findings` are categories only, never the text that triggered them."""

    text: str
    findings: tuple[str, ...] = ()
    action: str = "none"  # "none", "redacted" or "withheld"


def _kept_in_evidence(value: str, evidence: str) -> bool:
    return bool(value) and value.lower() in evidence


def _redact_pii(text: str, evidence: str, found: set[str]) -> str:
    def sub(pattern: re.Pattern[str], token: str, category: str, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            if _kept_in_evidence(match.group(0), evidence):
                return match.group(0)
            found.add(category)
            return token

        return pattern.sub(repl, text)

    text = sub(_EMAIL, "[EMAIL]", "email", text)
    text = sub(_NRIC, "[ID NUMBER]", "id_number", text)

    def card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits) and not _kept_in_evidence(match.group(0), evidence):
            found.add("payment_card")
            return "[CARD]"
        return match.group(0)

    text = _CARD_CANDIDATE.sub(card, text)
    return sub(_PHONE, "[PHONE]", "phone", text)


def guard_output(answer: object, evidence: Iterable[str] = ()) -> OutputResult:
    """Check the final answer. `evidence` is the text of the manual passages the answer was built from."""
    if not isinstance(answer, str) or not answer.strip():
        return OutputResult(text=WITHHELD_MESSAGE, findings=("empty_answer",), action="withheld")

    found: set[str] = set()
    text = answer
    evidence_text = "\n".join(evidence).lower()

    # 4 first, because a leaked prompt is withheld whole and there is no point cleaning it.
    lowered = " ".join(text.lower().split())
    if any(signature in lowered for signature in PROMPT_SIGNATURES):
        found.add("prompt_leak")
    if _INTERNAL_JSON.match(text):
        found.add("internal_structure")
    if found:
        return OutputResult(text=WITHHELD_MESSAGE, findings=tuple(sorted(found)), action="withheld")

    # 1. secrets
    for pattern, token in _HARD_SECRETS:
        text, count = pattern.subn(token, text)
        if count:
            found.add("secret")

    def credential(match: re.Match[str]) -> str:
        if _kept_in_evidence(match.group(0), evidence_text):
            return match.group(0)
        found.add("secret")
        return f"{match.group(1)}=[REDACTED]"

    text = _CREDENTIAL_ASSIGNMENT.sub(credential, text)

    # 2. personal data
    text = _redact_pii(text, evidence_text, found)

    # 3. markup
    for pattern in _MARKUP:
        text, count = pattern.subn("[removed]", text)
        if count:
            found.add("markup")

    # 5. length
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rstrip() + TRUNCATION_NOTICE
        found.add("too_long")

    return OutputResult(text=text, findings=tuple(sorted(found)), action="redacted" if found else "none")
