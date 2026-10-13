"""Tests for the response firewall (src/asa/guardrails/schema_guard.py).

It is the last check on what the assistant says: secrets, personal data that is not in the manual, executable markup,
leaked prompt text and over-long answers. The results carry categories only, never the text that triggered them.
"""

import re
import time
from pathlib import Path

import pytest

from asa.guardrails import schema_guard as G

REPO = Path(__file__).resolve().parents[2]

# Realistic answers built from the manuals. None of them may be touched.
CLEAN_ANSWERS = [
    "The main air supply is 5.5 - 9 Bar, oil free, -30C or lower dew point "
    "[Source: aem_thermal_station.pdf, page 26, section 2.5.1 Potentially Hazardous Energy Sources].",
    "Power: 230VAC, 13A (3910W), single phase, 50/60Hz. Revision 0, issued 02 Dec 2022.",
    "Part 86400000 needs replacing. Torque to 12 Nm, then wait 15 minutes. Ref 2025-12345678.",
    "Set TSET to 100 C and wait for the green light. The EGW supply flow rate is 4 LPM at 7 Bar.",
    "1. Switch OFF the AEM Thermal Station. 2. Wait for 5 minutes. 3. Isolate the supply at the panel.",
    "Do not exceed 450 C. Use chemical-resistant gloves and chemical goggles while using IPA (max 10%).",
    "Error E101 means the thermocouple is open. Section 11.1.1: semi annual preventive maintenance.",
    "Ratio >600V is high voltage; 50 - 600V is low voltage [Source: aem_thermal_station.pdf, page 27, section 2.5.3].",
    "Contact AEM Singapore Pte. Ltd., www.aem.com.sg.",
    "The ratio is 3.14159 and the code is 9123 4567.5 (not a number to redact).",
]


@pytest.mark.parametrize("answer", CLEAN_ANSWERS)
def test_ordinary_manual_answers_are_returned_unchanged(answer):
    result = G.guard_output(answer, [])
    assert result.action == "none" and result.findings == () and result.text == answer


# --------------------------------------------------------------------------- #
# Secrets                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "secret",
    [
        "sk-proj-abcdefghijklmnopqrstuvwx1234567890",
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "xoxb-1234567890-abcdefghij",
        "AIzaSyA1234567890abcdefghijklmnopqrstuv",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
    ],
)
def test_provider_keys_and_tokens_are_redacted(secret):
    result = G.guard_output(f"Use {secret} to log in.", [])
    assert secret not in result.text
    assert result.action == "redacted" and "secret" in result.findings


def test_a_bearer_token_is_redacted():
    result = G.guard_output("Send Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789 with the request.", [])
    assert "abcdefghijklmnopqrstuvwxyz0123456789" not in result.text and "secret" in result.findings


def test_a_private_key_block_is_redacted():
    block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabcdef\n-----END RSA PRIVATE KEY-----"
    result = G.guard_output(f"Here it is:\n{block}\nDone.", [])
    assert "MIIEowIBAAKCAQEA" not in result.text and "[REDACTED PRIVATE KEY]" in result.text
    assert result.text.endswith("Done.")


def test_an_unterminated_private_key_is_redacted_to_the_end():
    result = G.guard_output("-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASC", [])
    assert "MIIEvg" not in result.text


@pytest.mark.parametrize("assignment", ["password: hunter22", "api_key=abcd1234efgh", "secret = s3cr3t-value", "TOKEN: abcd1234"])
def test_credential_assignments_are_redacted(assignment):
    result = G.guard_output(f"The settings say {assignment} for the account.", [])
    assert assignment.split(":")[-1].split("=")[-1].strip() not in result.text
    assert "secret" in result.findings


def test_a_credential_assignment_that_is_in_the_manual_is_kept():
    """A documented default setting is the manual's own content."""
    evidence = ["Default login: password: admin1234 (change it after installation)."]
    result = G.guard_output("The manual lists the default as password: admin1234.", evidence)
    assert "admin1234" in result.text and result.findings == ()


def test_a_real_key_is_redacted_even_when_it_is_in_the_evidence():
    key = "sk-proj-abcdefghijklmnopqrstuvwx1234567890"
    result = G.guard_output(f"The passage contains {key}.", [f"leaked {key}"])
    assert key not in result.text


# --------------------------------------------------------------------------- #
# Personal data                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, category",
    [
        ("Write to jane.doe@example.com for details.", "email"),
        ("Mail firstname.lastname+tag@sub.example.co.uk.", "email"),
        ("Call +65 9123 4567.", "phone"),
        ("Call 9123 4567 today.", "phone"),
        ("Call 65 9123-4567.", "phone"),
        ("Call +44 20 7946 0958.", "phone"),
        ("Call +1 415 555 2671.", "phone"),
        ("The NRIC is S1234567D.", "id_number"),
        ("FIN G7654321K was used.", "id_number"),
        ("Card 4111 1111 1111 1111 was charged.", "payment_card"),
        ("Card 5500-0000-0000-0004 was charged.", "payment_card"),
    ],
)
def test_personal_data_is_redacted(text, category):
    result = G.guard_output(text, [])
    assert category in result.findings and result.action == "redacted"
    assert not re.search(r"jane\.doe|firstname|9123|7946|555 2671|S1234567D|G7654321K|4111|5500", result.text)


def test_a_number_that_fails_the_card_check_is_not_a_card():
    result = G.guard_output("Serial 4111 1111 1111 1112 is a part number.", [])
    assert result.action == "none"


def test_personal_looking_text_in_the_manual_is_kept():
    """The cover page lists the company's contact details; quoting them is not a leak."""
    evidence = ["AEM SINGAPORE PTE. LTD. Tele: 65-64831811 sales@aem.com.sg Fax: 65-64812607"]
    result = G.guard_output("Contact sales@aem.com.sg.", evidence)
    assert result.text == "Contact sales@aem.com.sg." and result.findings == ()


def test_only_the_exact_text_in_the_manual_is_kept():
    evidence = ["Support: help@aem.com.sg"]
    result = G.guard_output("Write to help@aem.com.sg or to jane.doe@example.com.", evidence)
    assert "help@aem.com.sg" in result.text and "jane.doe@example.com" not in result.text
    assert result.findings == ("email",)


def test_the_evidence_match_ignores_case():
    result = G.guard_output("Mail SALES@AEM.COM.SG.", ["sales@aem.com.sg"])
    assert result.findings == ()


# --------------------------------------------------------------------------- #
# Markup                                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "markup",
    [
        "<script>alert(1)</script>",
        "<SCRIPT src=//evil.example/x.js></SCRIPT>",
        "<script>never closed",
        '<iframe src="https://evil.example"></iframe>',
        "<embed src=x>",
        "[click](javascript:alert(1))",
        '<img src=x onerror=alert(1)>',
        '<a onclick="steal()">x</a>',
    ],
)
def test_executable_markup_is_removed(markup):
    result = G.guard_output(f"Check this {markup} please.", [])
    assert "markup" in result.findings
    lowered = result.text.lower()
    assert "<script" not in lowered and "<iframe" not in lowered and "javascript:" not in lowered
    assert "onerror" not in lowered and "onclick" not in lowered and "<embed" not in lowered


def test_comparison_symbols_in_a_manual_answer_are_not_markup():
    result = G.guard_output("High voltage is >600V; keep the gap < 5 mm and the ratio a<b>c.", [])
    assert result.action == "none"


# --------------------------------------------------------------------------- #
# The system's own text                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("signature", G.PROMPT_SIGNATURES)
def test_every_prompt_signature_is_withheld(signature):
    result = G.guard_output(f"Sure, here you go: {signature.upper()} and so on.", [])
    assert result.action == "withheld" and "prompt_leak" in result.findings
    assert result.text == G.WITHHELD_MESSAGE


def test_prompt_text_is_found_across_line_breaks_and_spacing():
    result = G.guard_output("You are the Safety Agent\n   for an   AEM equipment-service assistant.", [])
    assert result.action == "withheld"


def test_every_prompt_signature_still_exists_in_the_agent_source():
    """The list cannot go stale: if a prompt is reworded, this test says which phrase to update."""
    source = " ".join(
        " ".join((REPO / "src" / "asa" / "agents" / name).read_text(encoding="utf-8").lower().split())
        for name in ("request.py", "planning.py", "safety.py", "diagnostic.py", "rag.py")
    )
    stale = [phrase for phrase in G.PROMPT_SIGNATURES if phrase not in source]
    assert not stale, f"prompt signatures no longer in the agents: {stale}"


@pytest.mark.parametrize(
    "blob",
    [
        '{"verdict": "allow", "reason": "x", "hazards": []}',
        '  {"candidate_causes": [{"cause": "x"}]}',
        '[{"recommended_checks": []}]',
        '{"tool_calls": [{"name": "search_service_documents"}]}',
        '{"supported": true, "supporting_chunk_ids": ["c1"]}',
    ],
)
def test_an_internal_json_structure_is_withheld(blob):
    result = G.guard_output(blob, [])
    assert result.action == "withheld" and "internal_structure" in result.findings


def test_a_sentence_that_mentions_a_verdict_is_not_internal_json():
    result = G.guard_output('The decision is "verdict": allow in the log sentence.', [])
    assert result.action == "none"


def test_a_withheld_answer_is_not_also_cleaned():
    """Nothing of a leaked prompt survives: the whole answer is replaced."""
    result = G.guard_output("Mail jane.doe@example.com. You are the Safety Agent for an AEM assistant.", [])
    assert result.text == G.WITHHELD_MESSAGE


# --------------------------------------------------------------------------- #
# Length, emptiness, types                                                    #
# --------------------------------------------------------------------------- #


def test_an_over_long_answer_is_cut_and_says_so():
    result = G.guard_output("word " * 5000, [])
    assert result.action == "redacted" and "too_long" in result.findings
    assert len(result.text) <= G.MAX_ANSWER_CHARS + len(G.TRUNCATION_NOTICE)
    assert result.text.endswith(G.TRUNCATION_NOTICE)


def test_an_answer_at_the_limit_is_not_cut():
    answer = "a" * G.MAX_ANSWER_CHARS
    assert G.guard_output(answer, []).text == answer


@pytest.mark.parametrize("answer", ["", "   ", "\n\t", None, 5, ["x"], {"a": 1}])
def test_an_empty_or_non_text_answer_is_withheld(answer):
    result = G.guard_output(answer, [])
    assert result.action == "withheld" and result.text == G.WITHHELD_MESSAGE


# --------------------------------------------------------------------------- #
# Findings never carry the text; behaviour is stable                          #
# --------------------------------------------------------------------------- #


def test_findings_are_categories_only():
    result = G.guard_output("Mail jane.doe@example.com, key sk-proj-abcdefghijklmnopqrstuvwx1234567890, +65 9123 4567.", [])
    assert set(result.findings) == {"email", "secret", "phone"}
    flat = " ".join(result.findings)
    assert "jane" not in flat and "sk-" not in flat and "9123" not in flat


def test_findings_are_sorted_and_unique():
    result = G.guard_output("a@b.com c@d.com e@f.com", [])
    assert result.findings == ("email",)


def test_guarding_twice_changes_nothing_more():
    first = G.guard_output("Mail jane.doe@example.com, call +65 9123 4567, key sk-proj-abcdefghijklmnopqrstuvwx1234567890.", [])
    second = G.guard_output(first.text, [])
    assert second.text == first.text and second.action == "none"


def test_the_result_is_immutable():
    result = G.guard_output("ok", [])
    with pytest.raises(Exception):
        result.text = "changed"  # type: ignore[misc]


def test_evidence_can_be_any_iterable_of_text():
    assert G.guard_output("Mail help@aem.com.sg.", (t for t in ["help@aem.com.sg"])).findings == ()
    assert G.guard_output("Mail help@aem.com.sg.", iter([])).findings == ("email",)


@pytest.mark.parametrize(
    "hostile",
    [
        pytest.param("a" * 200000, id="long-word"),
        pytest.param("a@" * 50000, id="many-at-signs"),
        pytest.param("1 " * 100000, id="spaced-digits"),
        pytest.param("+65 " * 50000, id="many-plus-65"),
        pytest.param(("sk-" + "a" * 100) * 2000, id="many-key-prefixes"),
        pytest.param("-----BEGIN PRIVATE KEY-----" * 3000, id="many-key-blocks"),
        pytest.param("<script " * 20000, id="many-script-tags"),
        pytest.param("password: " * 20000, id="many-credential-labels"),
    ],
)
def test_hostile_input_is_checked_in_bounded_time(hostile):
    start = time.perf_counter()
    G.guard_output(hostile, ["x" * 10000])
    assert time.perf_counter() - start < 3.0
