import json
import logging
import time
import urllib.error
import urllib.request

from app.config import settings

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "The assistant is temporarily unavailable. Please try again in a moment."


def _call_ollama(message: str) -> str:
    payload = json.dumps({"model": settings.model_name, "prompt": message, "stream": False}).encode("utf-8")

    req = urllib.request.Request(
        f"{settings.ollama_base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    response = urllib.request.urlopen(req, timeout=settings.ollama_timeout)
    data = json.loads(response.read().decode("utf-8"))
    return data.get("response", "")


def _call_ollama_with_retry(message: str) -> str:
    attempts = settings.ollama_max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return _call_ollama(message)
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            # Log the error type only: messages can echo the request URL.
            logger.warning(
                "Ollama call failed (attempt %d/%d): %s",
                attempt,
                attempts,
                type(exc).__name__,
            )
            if attempt < attempts:
                time.sleep(settings.ollama_retry_backoff * 2 ** (attempt - 1))
    return FALLBACK_MESSAGE


def generate_reply(message: str) -> str:
    if not message or not message.strip():
        return ""

    provider = settings.model_provider.lower()

    if provider == "ollama":
        return _call_ollama_with_retry(message)

    if provider == "openai":
        return "OpenAI provider selected; add integration logic here."

    return f"Unsupported provider: {provider}"
