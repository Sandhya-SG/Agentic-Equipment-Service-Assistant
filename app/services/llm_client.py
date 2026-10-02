import json
import os
import urllib.request


def generate_reply(message: str) -> str:
    if not message or not message.strip():
        return ""

    provider = os.getenv("MODEL_PROVIDER", "ollama").lower()
    model_name = os.getenv("MODEL_NAME", "llama3.1")

    if provider == "ollama":
        payload = json.dumps(
            {
                "model": model_name,
                "prompt": message,
                "stream": False,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            response = urllib.request.urlopen(req, timeout=30)
            body = response.read().decode("utf-8")
            data = json.loads(body)
            return data.get("response", "")
        except Exception:
            return f"Model provider '{provider}' is not available. Please ensure the service is running."

    if provider == "openai":
        return "OpenAI provider selected; add integration logic here."

    return f"Unsupported provider: {provider}"
