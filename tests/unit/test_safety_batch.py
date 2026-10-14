"""Grouped safety-claim validation: several claims share one model call, with a per-claim fallback.

The evidence passages are the bulk of every validation prompt. Sending them once for a group of claims instead of once
per claim cuts the tokens a lot; these tests pin that the answers are still read claim by claim, that nothing is
invented or lost, and that anything the group call does not answer is validated on its own as before.
"""

import json
import re
import threading
import time
from types import SimpleNamespace

import pytest

import asa.agents.safety as safety
from asa.graph.state import Chunk

CHUNK = Chunk(
    chunk_id="safety_p19",
    doc_id="aem_thermal_station",
    section_id="2.2.2",
    revision="0",
    equipment_model="thermal_station",
    text="Electrical Hazards: Hazardous voltages may exist. Power off before servicing.",
    score=1.0,
    source_file="aem_thermal_station.pdf",
    page=19,
    section_title="2.2.2 General Safety Guidelines",
)


def reply(payload) -> SimpleNamespace:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def numbered_claims(prompt: str) -> list[str]:
    """The claims of a group prompt, in order."""
    return re.findall(r"^\d+\. \[[A-Z_]+\] (.*)$", prompt, flags=re.M)


class Model:
    """Answers group prompts from `supported`, and single-claim prompts too; records every call."""

    def __init__(self, supported, group=None, single=None):
        self.supported = set(supported)
        self.group = group  # a function (claims) -> payload, to override the group answer
        self.single = single
        self.calls = []

    def __call__(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        grouped = "Candidate claims, each with a number" in prompt
        self.calls.append(("group" if grouped else "single", prompt))
        if grouped:
            claims = numbered_claims(prompt)
            if self.group:
                return reply(self.group(claims))
            return reply(
                {
                    "results": [
                        {
                            "claim": n,
                            "supported": c in self.supported,
                            "supporting_chunk_ids": [CHUNK.chunk_id] if c in self.supported else [],
                        }
                        for n, c in enumerate(claims, 1)
                    ]
                }
            )
        claim = prompt.split("Candidate claim:\n", 1)[1].split("\n", 1)[0]
        if self.single:
            return reply(self.single(claim))
        ok = claim in self.supported
        return reply({"supported": ok, "supporting_chunk_ids": [CHUNK.chunk_id] if ok else []})

    def count(self, kind):
        return sum(1 for k, _ in self.calls if k == kind)


@pytest.fixture
def model(monkeypatch):
    def install(supported, **kwargs):
        fake = Model(supported, **kwargs)
        monkeypatch.setattr(safety.client.chat.completions, "create", fake)
        return fake

    return install


def validate(hazards=(), ppe=(), controls=()):
    return safety._validate_safety_items(
        question="What precautions apply?",
        chunks=[CHUNK],
        hazards=list(hazards),
        ppe_required=list(ppe),
        required_controls=list(controls),
    )


# --- fewer calls, same answers ---


def test_claims_share_calls_in_groups_of_at_most_five(model):
    fake = model({"h0", "p0", "c1"})
    valid_hazards, valid_ppe, valid_controls, claims = validate(
        [f"h{i}" for i in range(4)], [f"p{i}" for i in range(3)], [f"c{i}" for i in range(4)]
    )
    # 11 claims: two groups of 5 share a call each, and the one left over is checked with the single-claim prompt
    assert fake.count("group") == 2 and fake.count("single") == 1
    assert valid_hazards == ["h0"] and valid_ppe == ["p0"] and valid_controls == ["c1"]
    assert [c.text for c in claims] == ["h0", "p0", "c1"]


def test_a_lone_claim_uses_the_single_claim_prompt(model):
    fake = model({"h0"})
    valid_hazards, *_ = validate(["h0"])
    assert fake.count("single") == 1 and fake.count("group") == 0
    assert valid_hazards == ["h0"]


def test_the_evidence_is_sent_once_per_group_not_once_per_claim(model):
    fake = model(set())
    validate([f"hazard {i}" for i in range(5)])
    assert fake.count("group") == 1
    assert fake.calls[0][1].count(CHUNK.text) == 1


def test_results_keep_the_order_and_category_of_every_claim(model):
    hazards = [f"hazard {i}" for i in range(5)]
    ppe = [f"ppe {i}" for i in range(4)]
    controls = [f"control {i}" for i in range(4)]
    model(set(hazards[::2] + ppe[1::2] + controls))
    valid_hazards, valid_ppe, valid_controls, claims = validate(hazards, ppe, controls)
    assert valid_hazards == ["hazard 0", "hazard 2", "hazard 4"]
    assert valid_ppe == ["ppe 1", "ppe 3"]
    assert valid_controls == controls
    assert [c.category for c in claims] == ["hazard"] * 3 + ["ppe"] * 2 + ["control"] * 4
    assert all(c.supporting_chunk_ids == [CHUNK.chunk_id] for c in claims)


# --- a group that is not answered properly falls back to one call per claim ---


def test_a_claim_missing_from_the_answer_is_checked_on_its_own(model):
    def answer_only_first(claims):
        return {"results": [{"claim": 1, "supported": True, "supporting_chunk_ids": [CHUNK.chunk_id]}]}

    fake = model({"a", "b", "c"}, group=answer_only_first)
    valid_hazards, *_ = validate(["a", "b", "c"])
    assert fake.count("group") == 1 and fake.count("single") == 2  # claims 2 and 3 were checked individually
    assert valid_hazards == ["a", "b", "c"]


@pytest.mark.parametrize(
    "payload",
    ["not json at all", {"results": "nope"}, {"other": []}, {"results": [None, 3, "x"]}, []],
)
def test_an_unreadable_group_answer_falls_back_for_every_claim(model, payload):
    fake = model({"a", "c"}, group=lambda claims: payload)
    valid_hazards, *_ = validate(["a", "b", "c"])
    assert fake.count("single") == 3
    assert valid_hazards == ["a", "c"]


def test_a_failing_group_call_falls_back_for_every_claim(model, monkeypatch):
    fake = Model({"a"})

    def fail_on_group(**kwargs):
        if "Candidate claims, each with a number" in kwargs["messages"][1]["content"]:
            raise RuntimeError("API unavailable")
        return fake(**kwargs)

    monkeypatch.setattr(safety.client.chat.completions, "create", fail_on_group)
    valid_hazards, *_ = validate(["a", "b"])
    assert valid_hazards == ["a"]


def test_when_everything_fails_no_claim_is_supported_and_nothing_crashes(monkeypatch):
    def always_fail(**kwargs):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(safety.client.chat.completions, "create", always_fail)
    assert validate(["a", "b"], ["p"], ["c"]) == ([], [], [], [])


# --- the answer is never trusted beyond the evidence ---


def test_invented_chunk_ids_are_discarded_and_a_claim_with_none_left_is_unsupported(model):
    def answer(claims):
        return {
            "results": [
                {"claim": 1, "supported": True, "supporting_chunk_ids": ["invented", CHUNK.chunk_id, CHUNK.chunk_id]},
                {"claim": 2, "supported": True, "supporting_chunk_ids": ["invented"]},
            ]
        }

    model(set(), group=answer)
    valid_hazards, _, _, claims = validate(["a", "b"])
    assert valid_hazards == ["a"]
    assert claims[0].supporting_chunk_ids == [CHUNK.chunk_id]


@pytest.mark.parametrize("supported", ["true", 1, None, "yes"])
def test_only_the_boolean_true_counts_as_supported(model, supported):
    answer = {
        "results": [{"claim": n, "supported": supported, "supporting_chunk_ids": [CHUNK.chunk_id]} for n in (1, 2)]
    }
    model(set(), group=lambda claims: answer)
    valid_hazards, *_ = validate(["a", "b"])
    assert valid_hazards == []


def test_claim_numbers_outside_the_group_are_ignored(model):
    answer = {
        "results": [
            {"claim": 0, "supported": True, "supporting_chunk_ids": [CHUNK.chunk_id]},
            {"claim": 9, "supported": True, "supporting_chunk_ids": [CHUNK.chunk_id]},
            {"claim": True, "supported": True, "supporting_chunk_ids": [CHUNK.chunk_id]},
            {"claim": "1", "supported": True, "supporting_chunk_ids": [CHUNK.chunk_id]},
        ]
    }
    fake = model(set(), group=lambda claims: answer)
    valid_hazards, *_ = validate(["a", "b"])
    assert valid_hazards == []  # nothing usable: both claims are then checked on their own (and not supported)
    assert fake.count("single") == 2


def test_the_first_answer_for_a_claim_counts(model):
    answer = {
        "results": [
            {"claim": 1, "supported": False, "supporting_chunk_ids": []},
            {"claim": 1, "supported": True, "supporting_chunk_ids": [CHUNK.chunk_id]},
            {"claim": 2, "supported": True, "supporting_chunk_ids": [CHUNK.chunk_id]},
        ]
    }
    model(set(), group=lambda claims: answer)
    valid_hazards, *_ = validate(["a", "b"])
    assert valid_hazards == ["b"]


# --- groups still run concurrently ---


def test_groups_run_concurrently(monkeypatch):
    inside, peak, lock = [], [], threading.Lock()
    fake = Model({f"h{i}" for i in range(10)})

    def slow(**kwargs):
        with lock:
            inside.append(1)
            peak.append(len(inside))
        time.sleep(0.2)
        with lock:
            inside.pop()
        return fake(**kwargs)

    monkeypatch.setattr(safety.client.chat.completions, "create", slow)
    started = time.monotonic()
    valid_hazards, *_ = validate([f"h{i}" for i in range(10)])
    assert time.monotonic() - started < 0.35  # two groups at the same time, not one after the other
    assert max(peak) == 2 and len(valid_hazards) == 10


# --- the saving itself ---


def test_the_group_prompt_is_much_smaller_than_one_prompt_per_claim(monkeypatch):
    """The claim: 10 claims cost about two evidence blocks instead of ten."""
    big = Chunk(**{**CHUNK.__dict__, "text": "Hazard text. " * 300})
    sizes = {"group": 0, "single": 0}

    def record(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        grouped = "Candidate claims, each with a number" in prompt
        sizes["group" if grouped else "single"] += len(prompt)
        return reply({"results": [{"claim": n, "supported": False, "supporting_chunk_ids": []} for n in range(1, 6)]})

    monkeypatch.setattr(safety.client.chat.completions, "create", record)
    safety._validate_safety_items(
        question="q", chunks=[big], hazards=[f"h{i}" for i in range(10)], ppe_required=[], required_controls=[]
    )
    per_claim_cost = len(big.text) * 10  # what ten single-claim prompts would carry for the evidence alone
    assert sizes["single"] == 0
    assert sizes["group"] < per_claim_cost / 3
