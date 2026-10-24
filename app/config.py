"""Leer variables de entorno (o de un archivo .env en local)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    # Usar producción por defecto: si falta configuración, fallar cerrado (seguro).
    # En local, poner ENVIRONMENT=local en el .env.
    environment: Literal["local", "production"] = "production"
    log_level: str = "INFO"

    # Key plataforma reto Banorte hacia endpoint
    agent_api_key: SecretStr | None = None

    # Keys LLMs
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    database_url: SecretStr | None = None

    # Perfil: fuera del repo. En Render, subirlo como Secret File (/etc/secrets/profile.yaml).
    profile_path: Path = Path("private/profile.yaml")

    # Agente: deducir el proveedor del nombre del modelo (claude-* o gpt-*).
    # Modelos elegidos con los evals (docs/decisions.md §5); el respaldo es de otro proveedor a propósito.
    llm_model: str = "claude-haiku-5-5"
    llm_fallback_model: str | None = "gpt-6.1-sol"
    llm_effort: Literal["low", "medium", "high"] = "low"
    context_strategy: Literal["completo", "herramientas"] = "completo"
    max_output_tokens: int = 8000
    llm_timeout_seconds: float = 60.0

    # Límites de entrada y de uso
    max_message_chars: int = 4000
    max_history_messages: int = 40
    rate_limit_per_minute: int = 30

    @model_validator(mode="after")
    def _require_key_in_production(self) -> "Settings":
        if self.environment == "production" and self.agent_api_key is None:
            raise ValueError("AGENT_API_KEY es obligatoria cuando ENVIRONMENT=production.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
