import json
import logging
import time
import urllib.error
import urllib.request

from app.config import settings
from asa.graph.state import GenerationResult, ModelUnavailable

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "The assistant is temporarily unavailable. Please try again in a moment."


def _call_ollama(message: str) -> GenerationResult:
    payload = json.dumps({"model": settings.model_name, "prompt": message, "stream": False}).encode("utf-8")

    req = urllib.request.Request(
        f"{settings.ollama_base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    response = urllib.request.urlopen(req, timeout=settings.ollama_timeout)
    data = json.loads(response.read().decode("utf-8"))
    return GenerationResult(
        text=data.get("response", ""),
        model=settings.model_name,
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
    )


def _call_ollama_with_retry(message: str) -> GenerationResult:
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
    raise ModelUnavailable("Ollama did not respond after retries")


def generate_with_usage(message: str) -> GenerationResult:
    """Call the configured model. Raises ModelUnavailable when it cannot be reached."""
    if not message or not message.strip():
        return GenerationResult(text="")

    provider = settings.model_provider.lower()

    if provider == "ollama":
        return _call_ollama_with_retry(message)

    if provider == "openai":
        return GenerationResult(text="OpenAI provider selected; add integration logic here.")

    return GenerationResult(text=f"Unsupported provider: {provider}")


def generate_reply(message: str) -> str:
    """Text-only variant. Returns a fallback message when the model is unavailable."""
    try:
        return generate_with_usage(message).text
    except ModelUnavailable:
        return FALLBACK_MESSAGE
