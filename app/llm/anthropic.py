"""Proveedor Claude (Anthropic Messages API) con streaming, herramientas y prompt caching."""

from collections.abc import AsyncIterator

import anthropic

from app.llm.base import (
    MAX_TOOL_ROUNDS,
    Completed,
    Event,
    LLMError,
    Message,
    SystemPrompt,
    TextDelta,
    Tool,
    ToolCalled,
    Usage,
)

# Usar el fallback del lado del servidor en Opus 5.5 y Sonnet 5.5: si un clasificador de seguridad
# rechaza la petición, la API la reintenta con el modelo que Anthropic recomienda para esa categoría.
# Haiku 5.5 no lo soporta.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Reintentar la ronda si el modelo genera JSON de herramienta que el SDK no puede leer.
_MAX_JSON_RETRIES = 2


class AnthropicProvider:
    def __init__(self, api_key: str, timeout: float):
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=2)

    async def run(
        self,
        *,
        model: str,
        system: SystemPrompt,
        messages: list[Message],
        tools: list[Tool],
        effort: str,
        max_output_tokens: int,
    ) -> AsyncIterator[Event]:
        # Marcar el bloque estable (reglas + perfil) con cache_control: se cobra a ~10% en cada
        # request siguiente. Poner el bloque dinámico después para no invalidar ese caché.
        system_blocks = [{"type": "text", "text": system.stable, "cache_control": {"type": "ephemeral"}}]
        if system.dynamic:
            system_blocks.append({"type": "text", "text": system.dynamic})

        tools_by_name = {t.name: t for t in tools}
        api_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.schema(),
                # Recibir los argumentos en cuanto se generan; Tool.call se encarga de validarlos.
                "eager_input_streaming": True,
            }
            for t in tools
        ]
        api_messages: list[dict] = [{"role": m.role, "content": m.content} for m in messages]

        usage = Usage()
        final = None
        json_retries = 0
        rounds = 0
        while rounds < MAX_TOOL_ROUNDS:
            request = {
                "model": model,
                "max_tokens": max_output_tokens,
                "system": system_blocks,
                "messages": api_messages,
                "output_config": {"effort": effort},
            }
            if api_tools:
                request["tools"] = api_tools
            emitted_text = False
            try:
                async with self._stream(request) as stream:
                    async for event in stream:
                        if event.type == "text":
                            emitted_text = True
                            yield TextDelta(event.text)
                    final = await stream.get_final_message()
            except ValueError:
                # JSON de herramienta ilegible: no hay tool_use_id que responder, así que repetir la ronda.
                json_retries += 1
                if json_retries > _MAX_JSON_RETRIES:
                    raise LLMError("El modelo generó argumentos de herramienta inválidos.") from None
                continue
            except anthropic.APIError as e:
                raise LLMError(f"Anthropic: {e}") from e

            rounds += 1
            json_retries = 0
            usage += _usage(final.usage)

            tool_uses = [b for b in final.content if b.type == "tool_use"]
            if final.stop_reason != "tool_use" or not tool_uses:
                break

            results = []
            for block in tool_uses:
                tool = tools_by_name.get(block.name)
                output, is_error = tool.call(block.input) if tool else (f"Herramienta desconocida: {block.name}", True)
                yield ToolCalled(name=block.name, arguments=block.input, is_error=is_error)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
                )
            # Historial append-only: devolver el turno completo del asistente (incluidos sus bloques
            # de thinking) sin cambios, como pide la API.
            api_messages.append({"role": "assistant", "content": final.content})
            api_messages.append({"role": "user", "content": results})
            if emitted_text:
                yield TextDelta("\n\n")  # separar el preámbulo del texto de la siguiente ronda

        yield Completed(
            model=final.model if final else model,
            usage=usage,
            stop_reason=final.stop_reason if final else "error",
        )

    def _stream(self, request: dict):
        if request["model"] in _FALLBACK_MODELS:
            return self._client.beta.messages.stream(**request, betas=[_FALLBACK_BETA], fallbacks="default")
        return self._client.messages.stream(**request)


def _usage(u) -> Usage:
    cached = u.cache_read_input_tokens or 0
    written = u.cache_creation_input_tokens or 0
    return Usage(
        input_tokens=u.input_tokens + cached + written,
        cached_tokens=cached,
        cache_write_tokens=written,
        output_tokens=u.output_tokens,
    )
