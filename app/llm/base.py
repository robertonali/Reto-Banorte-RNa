"""Interfaz común de los proveedores de LLM (Claude y GPT).

Definir los únicos tipos que conoce el agente. Cada proveedor traduce a su API y ejecuta el loop de
herramientas con su formato nativo, porque los dos manejan distinto el estado entre llamadas
(bloques de thinking en Claude, reasoning items en OpenAI).
"""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

# Máximo de rondas de herramientas por respuesta, para evitar loops infinitos y costos sorpresa.
MAX_TOOL_ROUNDS = 4


class LLMError(Exception):
    """Falla del proveedor (red, cuota, error de la API). El endpoint la reporta como model_error."""


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class SystemPrompt:
    # `stable` se cachea en el proveedor; `dynamic` (fecha, preferencias del cliente) va después.
    stable: str
    dynamic: str = ""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]  # para validar los argumentos que manda el modelo
    run: Callable[[BaseModel], str]

    def schema(self) -> dict:
        return self.args.model_json_schema()

    def call(self, raw_args: object) -> tuple[str, bool]:
        """Ejecutar la herramienta y devolver (resultado, es_error)."""
        try:
            args = self.args.model_validate(raw_args)
        except ValueError as e:
            return f"Argumentos inválidos: {e}", True
        return self.run(args), False


@dataclass
class Usage:
    input_tokens: int = 0  # total de entrada, incluye los tokens leídos de caché
    cached_tokens: int = 0  # leídos de caché (más baratos)
    cache_write_tokens: int = 0  # escritos a caché (solo Claude)
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


# Eventos que emite un proveedor mientras genera.
@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCalled:
    name: str
    arguments: object
    is_error: bool


@dataclass(frozen=True)
class Completed:
    model: str
    usage: Usage
    stop_reason: str


Event = TextDelta | ToolCalled | Completed


class LLMProvider(Protocol):
    def run(
        self,
        *,
        model: str,
        system: SystemPrompt,
        messages: list[Message],
        tools: list[Tool],
        effort: str,
        max_output_tokens: int,
    ) -> AsyncIterator[Event]: ...


# ─── Precios (USD por millón de tokens: entrada, entrada desde caché, salida) ───
# Fuentes: tabla de modelos de la skill claude-api (2026-10) y developers.openai.com/api/docs/pricing.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-sonnet-5-5": (2.00, 0.20, 10.00),
    "claude-haiku-5-5": (0.10, 0.01, 0.50),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}
CACHE_WRITE_MULTIPLIER = 1.25  # Claude cobra la escritura a caché a 1.25x la entrada


def cost_usd(model: str, usage: Usage) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_cached, price_out = PRICES[model]
    uncached = usage.input_tokens - usage.cached_tokens - usage.cache_write_tokens
    total = (
        uncached * price_in
        + usage.cached_tokens * price_cached
        + usage.cache_write_tokens * price_in * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * price_out
    )
    return total / 1_000_000
