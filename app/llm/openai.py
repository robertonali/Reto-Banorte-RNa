"""Proveedor GPT (OpenAI Responses API) con streaming y herramientas."""

import json
from collections.abc import AsyncIterator

import openai

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


class OpenAIProvider:
    def __init__(self, api_key: str, timeout: float):
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=2)

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
        # OpenAI cachea solo prefijos repetidos: poner lo estable primero y lo dinámico al final.
        instructions = system.stable + (f"\n\n{system.dynamic}" if system.dynamic else "")
        tools_by_name = {t.name: t for t in tools}
        api_tools = [
            {"type": "function", "name": t.name, "description": t.description, "parameters": t.schema(), "strict": True}
            for t in tools
        ]
        input_items: list = [{"role": m.role, "content": m.content} for m in messages]

        usage = Usage()
        final = None
        for _ in range(MAX_TOOL_ROUNDS):
            request = {
                "model": model,
                "instructions": instructions,
                "input": input_items,
                "reasoning": {"effort": effort},
                "max_output_tokens": max_output_tokens,
                # No guardar conversaciones en OpenAI; el razonamiento viaja cifrado entre rondas.
                "store": False,
                "include": ["reasoning.encrypted_content"],
                "prompt_cache_key": "agente-cv",
            }
            if api_tools:
                request["tools"] = api_tools
            final = None
            emitted_text = False
            try:
                stream = await self._client.responses.create(**request, stream=True)
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        emitted_text = True
                        yield TextDelta(event.delta)
                    elif event.type in ("response.completed", "response.incomplete"):
                        final = event.response
                    elif event.type == "response.failed":
                        raise LLMError(f"OpenAI: {event.response.error}")
                    elif event.type == "error":
                        raise LLMError(f"OpenAI: {event.message}")
            except openai.APIError as e:
                raise LLMError(f"OpenAI: {e}") from e
            if final is None:
                raise LLMError("OpenAI: el stream terminó sin respuesta final.")

            usage += _usage(final.usage)
            calls = [item for item in final.output if item.type == "function_call"]
            if not calls:
                break

            # Devolver todos los items de la ronda (incluido el razonamiento) más los resultados.
            input_items += final.output
            for call in calls:
                try:
                    raw_args = json.loads(call.arguments)
                except json.JSONDecodeError:
                    raw_args = None
                tool = tools_by_name.get(call.name)
                output, is_error = tool.call(raw_args) if tool else (f"Herramienta desconocida: {call.name}", True)
                yield ToolCalled(name=call.name, arguments=raw_args, is_error=is_error)
                input_items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
            if emitted_text:
                yield TextDelta("\n\n")

        yield Completed(
            model=final.model if final else model,
            usage=usage,
            stop_reason=final.status if final else "error",
        )


def _usage(u) -> Usage:
    if u is None:
        return Usage()
    return Usage(
        input_tokens=u.input_tokens,
        cached_tokens=u.input_tokens_details.cached_tokens if u.input_tokens_details else 0,
        output_tokens=u.output_tokens,
        reasoning_tokens=u.output_tokens_details.reasoning_tokens if u.output_tokens_details else 0,
    )
