"""Safety rules — the deterministic layer of the Safety & Compliance Agent.

THIS IS A SAFETY ARTIFACT. The trigger lists below govern what causes the system
to halt a field engineer's proposed action. Treat this file as reviewed,
version-controlled, and (ideally) signed off by someone with domain knowledge of
the actual equipment. Expand it against AEM's real safety documentation.

Design principle (from the architecture): DETERMINISTIC RULES RUN FIRST and their
halts are FINAL. The LLM layer of the Safety agent runs afterwards and may only
ADD caution (more hazards, escalation) — it can never clear a hazard the rules
flagged. This is because a probabilistic model must not be trusted to decide
life-safety; hard hazards are matched by explicit, predictable rules.

Seeded from the synthetic XR-500 safety document (high voltage, ESD, stored
energy, thermal, lock-out/tag-out). Add real hazards as the corpus grows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from asa.graph.state import Step


# --------------------------------------------------------------------------- #
# Hard-halt triggers — matching any of these forces an execution HALT          #
# --------------------------------------------------------------------------- #
# Each hazard category maps to (a) detection patterns and (b) required PPE.
# Keep patterns specific enough to avoid over-triggering on benign text, but
# broad enough to catch real phrasings.

HARD_HALT_TRIGGERS: dict[str, list[str]] = {
    "high_voltage": [
        r"\bhigh\s+voltage\b",
        r"\bHV\b",
        r"\b\d{3,}\s?V(?:olts?)?\b",     # 3+ digit voltages, e.g. 480V
        r"\bmains\b",
        r"\bpower\s+distribution\b",
        r"\benergized\s+circuit",
    ],
    "stored_energy": [
        r"\bcapacitors?\b",
        r"\bstored\s+energy\b",
        r"\bdischarge\b",
        r"\bpressuri[sz]ed\b",
    ],
    "esd": [
        r"\bESD\b",
        r"\belectrostatic\b",
    ],
    "lockout_tagout": [
        r"\bLOTO\b",
        r"\block[\s-]?out\b",
        r"\btag[\s-]?out\b",
        r"\bde[\s-]?energi[sz]e",
    ],
    "thermal": [
        r"\bhot\s+surface",
        r"\bburn\s+hazard\b",
        r"\b\d{3,}\s?°?\s?C\b",           # 3+ digit temperatures
        r"\bheater\s+element\b",
    ],
}

# --------------------------------------------------------------------------- #
# PPE requirements per hazard                                                 #
# --------------------------------------------------------------------------- #

PPE_RULES: dict[str, list[str]] = {
    "high_voltage": ["insulated gloves", "arc-flash face shield"],
    "stored_energy": ["safety glasses", "insulated tools"],
    "esd": ["ESD wrist strap", "ESD-safe mat"],
    "lockout_tagout": ["lockout device", "personal lock and tag"],
    "thermal": ["heat-resistant gloves", "safety glasses"],
}

# Hazards that are NOT field-serviceable at all — these always escalate, even
# beyond a halt. Working on them requires a qualified engineer.
ESCALATE_ONLY_HAZARDS = {"high_voltage", "stored_energy"}


# --------------------------------------------------------------------------- #
# Result                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class SafetyRuleResult:
    """Outcome of the deterministic safety check."""
    forced_halt: bool
    hazards: list[str] = field(default_factory=list)
    ppe_required: list[str] = field(default_factory=list)
    escalate_only: bool = False          # hazard is not field-serviceable
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The deterministic check                                                     #
# --------------------------------------------------------------------------- #

_COMPILED_TRIGGERS = {
    hazard: [re.compile(p, re.IGNORECASE) for p in patterns]
    for hazard, patterns in HARD_HALT_TRIGGERS.items()
}


def _scan(text: str) -> tuple[list[str], list[str]]:
    """Return (hazards_found, matched_phrases) for a block of text."""
    hazards: list[str] = []
    matched: list[str] = []
    for hazard, patterns in _COMPILED_TRIGGERS.items():
        for pat in patterns:
            m = pat.search(text)
            if m:
                if hazard not in hazards:
                    hazards.append(hazard)
                matched.append(m.group(0).strip())
    return hazards, matched


def check_steps(steps: list[Step]) -> SafetyRuleResult:
    """Run the deterministic safety check over proposed troubleshooting steps.

    This is Layer 1 of the Safety agent. If ANY hard trigger matches, forced_halt
    is True and that decision is final — the LLM layer cannot override it.
    """
    combined = " ".join(s.action for s in steps)
    return check_text(combined)


def check_text(text: str) -> SafetyRuleResult:
    """Run the deterministic safety check over arbitrary text (steps or a query)."""
    hazards, matched = _scan(text)
    if not hazards:
        return SafetyRuleResult(forced_halt=False)

    ppe: list[str] = []
    for h in hazards:
        ppe.extend(PPE_RULES.get(h, []))

    escalate_only = any(h in ESCALATE_ONLY_HAZARDS for h in hazards)
    reasons = [f"deterministic trigger: {h}" for h in hazards]

    return SafetyRuleResult(
        forced_halt=True,
        hazards=hazards,
        ppe_required=sorted(set(ppe)),
        escalate_only=escalate_only,
        reasons=reasons,
    )
