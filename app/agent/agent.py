"""Agente de CV: armar el prompt, aplicar guardrails y delegar la generación al proveedor."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date

from app.agent import guardrails
from app.agent.prompts import Strategy, build_system
from app.agent.tools import build_tools
from app.config import Settings
from app.llm.anthropic import AnthropicProvider
from app.llm.base import Completed, Event, LLMError, LLMProvider, Message, SystemPrompt, TextDelta, ToolCalled
from app.llm.openai import OpenAIProvider
from app.profile import Profile

logger = logging.getLogger("cv_agent.agent")

_REFUSAL_TEXT = (
    "Lo siento, no puedo ayudar con esa solicitud. Con gusto te cuento sobre la experiencia, "
    "habilidades o proyectos de {nombre}."
)


@dataclass(frozen=True)
class AgentConfig:
    model: str
    strategy: Strategy
    effort: str
    max_output_tokens: int


@dataclass(frozen=True)
class Fallback:
    """Modelo de otro proveedor que responde si el principal falla (caída, cuota, timeout)."""

    provider: LLMProvider
    model: str


class Agent:
    def __init__(self, profile: Profile, provider: LLMProvider, config: AgentConfig, fallback: Fallback | None = None):
        self.profile = profile
        self.provider = provider
        self.config = config
        self.fallback = fallback
        self._tools = build_tools(profile) if config.strategy == "herramientas" else []

    async def stream(self, messages: list[Message], client_instructions: str | None = None) -> AsyncIterator[Event]:
        """Generar la respuesta como eventos (texto incremental, herramientas usadas y cierre)."""
        signals = guardrails.injection_signals(messages[-1].content)
        if signals:
            logger.warning("Posible prompt injection: %s", ",".join(signals))

        system = build_system(
            self.profile,
            self.config.strategy,
            today=date.today(),
            client_instructions=client_instructions,
            injection_warning=bool(signals),
        )
        emitted_text = False
        try:
            async for event in self._generate(self.provider, self.config.model, system, messages):
                emitted_text = emitted_text or isinstance(event, TextDelta)
                yield event
        except LLMError as e:
            # Cambiar de proveedor solo si el usuario todavía no recibió texto.
            if emitted_text or self.fallback is None:
                raise
            logger.warning("Falla de %s (%s); respondiendo con %s", self.config.model, e, self.fallback.model)
            async for event in self._generate(self.fallback.provider, self.fallback.model, system, messages):
                yield event

    async def _generate(
        self, provider: LLMProvider, model: str, system: SystemPrompt, messages: list[Message]
    ) -> AsyncIterator[Event]:
        emitted_text = False
        async for event in provider.run(
            model=model,
            system=system,
            messages=messages,
            tools=self._tools,
            effort=self.config.effort,
            max_output_tokens=self.config.max_output_tokens,
        ):
            if isinstance(event, TextDelta):
                emitted_text = True
            elif isinstance(event, ToolCalled):
                logger.info("Herramienta %s args=%s error=%s", event.name, event.arguments, event.is_error)
            elif isinstance(event, Completed) and event.stop_reason == "refusal" and not emitted_text:
                # Si un clasificador de seguridad rechazó la petición sin generar texto, responder amable.
                yield TextDelta(_REFUSAL_TEXT.format(nombre=self.profile.nombre.split()[0]))
            yield event


def build_provider(model: str, settings: Settings) -> LLMProvider:
    """Elegir el proveedor según el nombre del modelo."""
    if model.startswith("claude"):
        if settings.anthropic_api_key is None:
            raise ValueError("Falta ANTHROPIC_API_KEY para usar un modelo de Claude.")
        return AnthropicProvider(settings.anthropic_api_key.get_secret_value(), settings.llm_timeout_seconds)
    if model.startswith("gpt-"):
        if settings.openai_api_key is None:
            raise ValueError("Falta OPENAI_API_KEY para usar un modelo de OpenAI.")
        return OpenAIProvider(settings.openai_api_key.get_secret_value(), settings.llm_timeout_seconds)
    raise ValueError(f"No sé qué proveedor usar para el modelo '{model}'.")


def build_agent(
    settings: Settings,
    profile: Profile,
    *,
    model: str | None = None,
    strategy: Strategy | None = None,
    with_fallback: bool = True,
) -> Agent:
    model = model or settings.llm_model
    config = AgentConfig(
        model=model,
        strategy=strategy or settings.context_strategy,
        effort=settings.llm_effort,
        max_output_tokens=settings.max_output_tokens,
    )
    fallback = None
    if with_fallback and settings.llm_fallback_model and settings.llm_fallback_model != model:
        try:
            fallback = Fallback(build_provider(settings.llm_fallback_model, settings), settings.llm_fallback_model)
        except ValueError as e:
            logger.warning("Sin modelo de respaldo: %s", e)
    return Agent(profile, build_provider(model, settings), config, fallback)
