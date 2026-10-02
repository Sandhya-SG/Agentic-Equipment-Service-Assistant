import re


def validate_user_input(user_text: str) -> dict:
    text = (user_text or "").strip()

    if not text:
        return {"allowed": False, "reason": "empty input", "sanitized": ""}

    prompt_injection_patterns = [
        r"ignore previous instructions",
        r"ignore all previous",
        r"system prompt",
        r"developer prompt",
        r"reveal.*prompt",
        r"override.*instructions",
    ]

    normalized = text.lower()
    for pattern in prompt_injection_patterns:
        if re.search(pattern, normalized):
            return {
                "allowed": False,
                "reason": "unsafe prompt injection pattern detected",
                "sanitized": "",
            }

    sanitized = text.replace("\r", " ").replace("\n", " ").strip()
    return {
        "allowed": True,
        "reason": "input accepted",
        "sanitized": sanitized,
    }
