"""Tests for the deterministic safety rules.

Run with: pytest tests/unit/test_safety_rules.py -v
"""

import pytest

from asa.graph.state import Step
from asa.guardrails import safety_rules as S

# --- Hazard detection (must force halt) ---

@pytest.mark.parametrize("action,expected", [
    ("Open the high voltage power distribution enclosure", "high_voltage"),
    ("Discharge the capacitors before servicing", "stored_energy"),
    ("Inspect the contactor pins for ESD damage", "esd"),
    ("Perform lockout tagout before opening", "lockout_tagout"),
    ("The heater element reaches 400C", "thermal"),
])
def test_hazards_force_halt(action, expected):
    r = S.check_steps([Step(1, action)])
    assert r.forced_halt is True
    assert expected in r.hazards


# --- PPE is attached ---

def test_ppe_attached_for_hazard():
    r = S.check_steps([Step(1, "work near high voltage")])
    assert "insulated gloves" in r.ppe_required


# --- Escalate-only classification ---

def test_high_voltage_is_escalate_only():
    r = S.check_steps([Step(1, "service the high voltage unit")])
    assert r.escalate_only is True


def test_esd_is_not_escalate_only():
    # ESD is field-serviceable with proper PPE
    r = S.check_steps([Step(1, "clean contactor with ESD strap")])
    assert r.escalate_only is False


# --- False-positive protection (benign steps must not halt) ---

@pytest.mark.parametrize("action", [
    "Enter Maintenance Mode from the operator console",
    "Replace the vacuum filter cartridge",
    "Run a grip self-test in Diagnostic Mode",
    "Inspect the nozzle tip for debris",
])
def test_benign_steps_do_not_halt(action):
    assert S.check_steps([Step(1, action)]).forced_halt is False


# --- Multi-hazard ---

def test_multiple_hazards_detected():
    r = S.check_steps([Step(1, "service the high voltage heater element at 350C")])
    assert "high_voltage" in r.hazards
    assert "thermal" in r.hazards


# --- Empty input is safe ---

def test_no_steps_no_halt():
    assert S.check_steps([]).forced_halt is False
