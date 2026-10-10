"""Shared state schema for the Agentic Equipment Service Assistant.

Every agent and component reads from and writes to AgentState as it flows through
the LangGraph graph. This is the central contract of the system — change it only
by team agreement, since all agents depend on its shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Literal, Optional, TypedDict

# --------------------------------------------------------------------------- #
# Supporting types                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    """A retrieved passage with full provenance."""

    chunk_id: str
    doc_id: str
    section_id: str
    revision: str
    equipment_model: str
    text: str
    score: float = 0.0

    # Citation / provenance metadata
    source_file: str = ""
    page: int = 0
    section_title: str = "Unknown"


@dataclass
class RankedCause:
    """A candidate root cause, ranked by likelihood, tied to its evidence."""

    cause: str
    likelihood: float
    supporting_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Step:
    """One troubleshooting action. Descriptive only — NO executable capability.

    The absence of any command/actuator field enforces the 'zero hardware
    execution authority' guardrail at the type level. The system advises;
    the human acts.
    """

    order: int
    action: str
    supporting_chunk_ids: list[str] = field(default_factory=list)
    hazard_flag: bool = False


@dataclass
class Citation:
    """A user-facing citation pointing to the exact supporting text."""

    doc_id: str
    section_id: str
    revision: str
    quote_span: str


@dataclass
class TraceEvent:
    """One entry in the audit trail."""

    agent: str
    action: str
    timestamp: str
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The shared state                                                            #
# --------------------------------------------------------------------------- #


class AgentState(TypedDict, total=False):
    """The object passed node-to-node through the graph.

    `total=False` means not every key must be present at every step — each agent
    populates its own portion and RETURNS partial updates; LangGraph merges them.
    """

    # --- Request Agent writes ---
    raw_query: str
    intent: str
    equipment_model: Optional[str]
    symptoms: list[str]

    request_status: Literal[
        "READY",
        "CLARIFY",
    ]

    clarification_needed: bool
    clarification_question: Optional[str]

    clarification_response: Optional[str]
    resolved_query: Optional[str]
    clarification_count: int

    # --- Planning Agent writes ---
    plan: list[str]
    current_step: str
    planning_reason: Optional[str]
    iteration_count: int

    # --- Agentic RAG Agent writes ---
    retrieval_strategy: Literal["vector", "bm25", "hybrid"]
    retrieved_chunks: list[Chunk]
    context_relevance: float
    sufficiency: bool
    retry_count: int

    # --- Diagnostic Agent writes ---
    root_causes: list[RankedCause]
    troubleshooting_steps: list[Step]

    # --- Safety Agent writes ---
    safety_verdict: Literal["allow", "halt", "escalate"]
    hazards: list[str]
    ppe_required: list[str]
    safety_reason: str

    # --- Explanation Module writes ---
    confidence: float
    citations: list[Citation]
    final_answer: str

    # --- Control flow ---
    escalated: bool
    escalation_reason: str
    terminate: bool

    # --- Audit (append-only via reducer) ---
    trace: Annotated[list[TraceEvent], add]


# --------------------------------------------------------------------------- #
# Guardrail caps — imported by the routing functions                          #
# --------------------------------------------------------------------------- #

MAX_ITERATIONS = 5
MAX_RETRIES = 3
MAX_CLARIFICATIONS = 2
CONFIDENCE_FLOOR = 0.70


def new_state(
    raw_query: str,
    equipment_model: Optional[str] = None,
) -> AgentState:
    """Factory for a fresh state at the start of a run."""

    return AgentState(
        raw_query=raw_query,
        equipment_model=equipment_model,

        symptoms=[],

        clarification_needed=False,
        clarification_question=None,
        clarification_response=None,
        resolved_query=None,
        clarification_count=0,

        planning_reason=None,

        plan=[],
        iteration_count=0,

        retrieved_chunks=[],
        context_relevance=0.0,
        sufficiency=False,
        retry_count=0,

        root_causes=[],
        troubleshooting_steps=[],

        safety_verdict="allow",
        hazards=[],
        ppe_required=[],

        confidence=0.0,
        citations=[],

        escalated=False,
        terminate=False,

        trace=[],
    )
