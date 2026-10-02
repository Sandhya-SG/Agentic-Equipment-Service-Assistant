import os
from functools import lru_cache


class Settings:
    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "Equipment Service Assistant")
        self.environment = os.getenv("ENVIRONMENT", "development")

        self.backend_host = os.getenv("BACKEND_HOST", "0.0.0.0")
        self.backend_port = int(os.getenv("BACKEND_PORT", "8000"))

        self.frontend_host = os.getenv("FRONTEND_HOST", "0.0.0.0")
        self.frontend_port = int(os.getenv("FRONTEND_PORT", "5000"))

        self.model_provider = os.getenv("MODEL_PROVIDER", "ollama")
        self.model_name = os.getenv("MODEL_NAME", "llama3.1")

        self.allowed_origins = os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:5000,http://127.0.0.1:5000",
        )

        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
