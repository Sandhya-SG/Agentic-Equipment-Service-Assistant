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


# --- Requests to defeat a safety control (always halt, before the agents) ---

@pytest.mark.parametrize(
    "request_text,control",
    [
        ("Can I bypass the safety interlock?", "interlock"),
        ("How do I defeat the door interlock?", "interlock"),
        ("Can I disable the emergency stop?", "emergency stop"),
        ("jumper out the E-stop so the machine keeps running", "emergency stop"),
        ("The interlock keeps tripping, can I override it?", "interlock"),
        ("Is it OK to tape down the safety switch?", "safety switch"),
        ("Can we deactivate the light curtain during setup?", "light curtain"),
        ("The door interlock was bypassed by maintenance, what now?", "interlock"),
    ],
)
def test_requests_to_defeat_a_safety_control_are_detected(request_text, control):
    assert S.check_defeat_request(request_text) == control


@pytest.mark.parametrize(
    "request_text",
    [
        "What does the safety interlock do?",
        "Where is the emergency stop button?",
        "How do I reset the emergency stop?",
        "What safety devices does the system have?",
        "Why does the interlock trip?",
        "How do I disable the alarm sound?",
        "Override the temperature setpoint to 25C",
        "What are the emotional factors?",
        # A defeat word and a control in different sentences is not a request to defeat it.
        "Disable the screen saver. The emergency stop is on the front panel.",
        "",
    ],
)
def test_questions_about_safety_controls_are_not_defeat_requests(request_text):
    assert S.check_defeat_request(request_text) is None


# --- Voltages as the AEM manuals write them ---

@pytest.mark.parametrize("text", ["415VAC supply", "480 VDC bus", "230 volts", "415V, 3 phase", "Install the 415VAC supply"])
def test_three_digit_voltages_are_high_voltage(text):
    result = S.check_text(text)
    assert "high_voltage" in result.hazards
    assert result.escalate_only is True


@pytest.mark.parametrize("text", ["24V control circuit", "12 VDC sensor", "5V logic"])
def test_low_voltages_are_not_high_voltage(text):
    assert "high_voltage" not in S.check_text(text).hazards
