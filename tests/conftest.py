"""Configuración común de los tests: perfil de ejemplo, auth activa y un LLM falso (sin costo)."""

import os

# Configurar antes de importar la app: los tests nunca usan el perfil real ni llaman a un LLM.
os.environ.update(
    {
        "ENVIRONMENT": "local",
        "AGENT_API_KEY": "clave-de-prueba",
        "PROFILE_PATH": "data/profile.example.yaml",
        "LLM_MODEL": "claude-haiku-5-5",
        "LLM_FALLBACK_MODEL": "",
        "RATE_LIMIT_PER_MINUTE": "1000",
        "DATABASE_URL": "",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import ratelimit  # noqa: E402
from app.agent.agent import Agent, AgentConfig  # noqa: E402
from app.api.responses import get_agent  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm.base import Completed, LLMError, TextDelta, Usage  # noqa: E402
from app.main import app  # noqa: E402
from app.profile import load_profile  # noqa: E402

AUTH = {"Authorization": "Bearer clave-de-prueba"}


class FakeProvider:
    """Responder un texto fijo en partes y registrar lo que recibió."""

    def __init__(self, parts=("Hola, ", "soy el ", "asistente."), fail=False):
        self.parts = parts
        self.fail = fail
        self.calls: list[dict] = []

    async def run(self, *, model, system, messages, tools, effort, max_output_tokens):
        self.calls.append({"model": model, "system": system, "messages": messages, "tools": tools})
        if self.fail:
            raise LLMError("falla simulada")
        for part in self.parts:
            yield TextDelta(part)
        yield Completed(
            model=model,
            usage=Usage(input_tokens=100, cached_tokens=80, output_tokens=10),
            stop_reason="end_turn",
        )


@pytest.fixture
def profile():
    return load_profile(get_settings().profile_path)


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture
def client(fake, profile, monkeypatch):
    monkeypatch.setattr(ratelimit, "_limiter", ratelimit.RateLimiter())
    agent = Agent(profile, fake, AgentConfig(model="claude-haiku-5-5", strategy="completo", effort="low", max_output_tokens=500))
    app.dependency_overrides[get_agent] = lambda: agent
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
