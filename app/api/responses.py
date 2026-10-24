"""POST /v1/responses: el endpoint Open Responses del agente (con y sin streaming)."""

import itertools
import json
import logging
import secrets
import time
from collections import Counter
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.agent import guardrails
from app.agent.agent import Agent, build_agent
from app.api.errors import OpenResponsesError
from app.api.schemas import (
    InputItem,
    InputTokensDetails,
    OutputMessage,
    OutputText,
    OutputTokensDetails,
    ResponseError,
    ResponseObject,
    ResponseRequest,
    Usage,
)
from app.auth import require_api_key
from app.config import Settings, get_settings
from app.llm.base import Completed, LLMError, Message, TextDelta, cost_usd
from app.llm.base import Usage as LLMUsage
from app.ratelimit import rate_limit
from app.store import ConversationStore

logger = logging.getLogger("cv_agent.api")

router = APIRouter()


# ─── Dependencias ───────────────────────────────────────────────────────────


def get_agent(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> Agent:
    # Construir el agente en el primer request y reutilizarlo (así se comparten los clientes HTTP de los SDKs).
    if getattr(request.app.state, "agent", None) is None:
        request.app.state.agent = build_agent(settings, request.app.state.profile)
    return request.app.state.agent


def get_store(request: Request) -> ConversationStore:
    return request.app.state.store


# ─── Traducción del request a mensajes ──────────────────────────────────────


def _text(content: str | list | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if part.text)


def normalize_input(items: str | list[InputItem] | None) -> tuple[list[Message], list[str]]:
    """Convertir `input` en mensajes de usuario/asistente e instrucciones del cliente (system/developer).

    Ignorar los elementos que no son mensajes (reasoning, function_call, imágenes...).
    """
    if items is None:
        return [], []
    if isinstance(items, str):
        return [Message("user", items)], []
    messages, instructions = [], []
    for item in items:
        if item.type != "message" or item.role is None:
            continue
        text = _text(item.content).strip()
        if not text:
            continue
        if item.role in ("system", "developer"):
            instructions.append(text)
        else:
            messages.append(Message(item.role, text))
    return messages, instructions


def _log_request_shape(body: ResponseRequest, request: Request) -> None:
    """Registrar la forma del request (nunca su contenido) para conocer el contrato real de la plataforma."""
    auth = request.headers.get("authorization", "")
    auth_scheme = (
        "bearer" if auth.lower().startswith("bearer ")
        else "authorization-otro" if auth
        else "x-api-key" if "x-api-key" in request.headers
        else "ninguno"
    )
    if isinstance(body.input, list):
        roles = dict(Counter(i.role or "-" for i in body.input))
        types = dict(Counter(i.type for i in body.input))
    else:
        roles = types = {}
    logger.info(
        "request stream=%s input=%s n_items=%s roles=%s types=%s previous_response_id=%s instructions=%s "
        "model=%r auth=%s user_agent=%r",
        body.stream,
        type(body.input).__name__,
        len(body.input) if isinstance(body.input, list) else "-",
        roles,
        types,
        body.previous_response_id is not None,
        body.instructions is not None,
        body.model,
        auth_scheme,
        request.headers.get("user-agent"),
    )


# ─── Construcción de la respuesta ───────────────────────────────────────────


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def _usage(u: LLMUsage) -> Usage:
    return Usage(
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        total_tokens=u.input_tokens + u.output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=u.cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=u.reasoning_tokens),
    )


class _Turn:
    """Guardar el estado de una respuesta en curso y armar el ResponseObject en cada etapa."""

    def __init__(self, body: ResponseRequest, agent: Agent):
        self.body = body
        self.response_id = _new_id("resp")
        self.message_id = _new_id("msg")
        self.created_at = int(time.time())
        self.model = agent.config.model

    def response(self, status: str, *, text: str | None = None, completed: Completed | None = None,
                 error: ResponseError | None = None) -> ResponseObject:
        output = []
        if text is not None:
            output = [OutputMessage(id=self.message_id, content=[OutputText(text=text)])]
        return ResponseObject(
            id=self.response_id,
            created_at=self.created_at,
            completed_at=int(time.time()) if status in ("completed", "failed", "incomplete") else None,
            status=status,
            model=completed.model if completed else self.model,
            previous_response_id=self.body.previous_response_id,
            instructions=self.body.instructions,
            output=output,
            error=error,
            usage=_usage(completed.usage) if completed else None,
            metadata=self.body.metadata or {},
        )


async def _remember(store: ConversationStore, turn: _Turn, messages: list[Message], text: str,
                    completed: Completed, started: float) -> None:
    latency_ms = round((time.perf_counter() - started) * 1000)
    cost = cost_usd(completed.model, completed.usage)
    meta = {
        "model": completed.model,
        "stop_reason": completed.stop_reason,
        "latency_ms": latency_ms,
        "input_tokens": completed.usage.input_tokens,
        "cached_tokens": completed.usage.cached_tokens,
        "output_tokens": completed.usage.output_tokens,
        "cost_usd": round(cost, 6) if cost is not None else None,
    }
    logger.info("response id=%s %s", turn.response_id, " ".join(f"{k}={v}" for k, v in meta.items()))
    try:
        await store.save(turn.response_id, [*messages, Message("assistant", text.strip())], meta)
    except Exception:
        # No fallar: el usuario ya recibió su respuesta; solo se pierde la posibilidad de continuarla.
        logger.exception("No se pudo guardar la respuesta %s", turn.response_id)


# ─── Endpoint ───────────────────────────────────────────────────────────────


@router.post(
    "/v1/responses",
    response_model=ResponseObject,
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
async def create_response(
    body: ResponseRequest,
    request: Request,
    agent: Annotated[Agent, Depends(get_agent)],
    store: Annotated[ConversationStore, Depends(get_store)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    _log_request_shape(body, request)

    history: list[Message] = []
    if body.previous_response_id:
        stored = await store.get(body.previous_response_id)
        if stored is None:
            raise OpenResponsesError(
                404,
                f"No existe la respuesta '{body.previous_response_id}'.",
                type="not_found",
                code="previous_response_not_found",
                param="previous_response_id",
            )
        history = stored

    new_messages, input_instructions = normalize_input(body.input)
    full_history = history + new_messages
    messages = guardrails.trim_history(full_history, settings.max_history_messages)
    if len(messages) < len(full_history):
        logger.info("Historial recortado: %d -> %d mensajes", len(full_history), len(messages))
    guardrails.validate_conversation(messages, max_message_chars=settings.max_message_chars)
    client_instructions = "\n\n".join(x for x in [body.instructions, *input_instructions] if x) or None

    turn = _Turn(body, agent)
    if body.stream:
        events = _stream(turn, agent, store, messages, client_instructions)
        return StreamingResponse(
            events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    started = time.perf_counter()
    parts: list[str] = []
    completed = None
    try:
        async for event in agent.stream(messages, client_instructions):
            if isinstance(event, TextDelta):
                parts.append(event.text)
            elif isinstance(event, Completed):
                completed = event
    except LLMError as e:
        logger.error("Falla del modelo: %s", e)
        raise OpenResponsesError(500, "El modelo no pudo generar una respuesta.", type="model_error", code="model_error") from e

    text = "".join(parts)
    await _remember(store, turn, messages, text, completed, started)
    return turn.response("completed", text=text, completed=completed)


async def _stream(
    turn: _Turn, agent: Agent, store: ConversationStore, messages: list[Message], client_instructions: str | None
) -> AsyncIterator[str]:
    """Emitir los eventos SSE en el orden que exige la spec para un mensaje de texto."""
    sequence = itertools.count()

    def sse(event_type: str, **payload) -> str:
        data = {"type": event_type, "sequence_number": next(sequence), **payload}
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def dump(model) -> dict:
        return model.model_dump(mode="json")

    ids = {"item_id": turn.message_id, "output_index": 0, "content_index": 0}
    started = time.perf_counter()

    yield sse("response.created", response=dump(turn.response("in_progress")))
    yield sse("response.in_progress", response=dump(turn.response("in_progress")))
    yield sse(
        "response.output_item.added",
        output_index=0,
        item=dump(OutputMessage(id=turn.message_id, status="in_progress", content=[])),
    )
    yield sse("response.content_part.added", **ids, part=dump(OutputText(text="")))

    parts: list[str] = []
    completed = None
    try:
        async for event in agent.stream(messages, client_instructions):
            if isinstance(event, TextDelta):
                parts.append(event.text)
                yield sse("response.output_text.delta", **ids, delta=event.text, logprobs=[])
            elif isinstance(event, Completed):
                completed = event
    except LLMError as e:
        logger.error("Falla del modelo durante el stream: %s", e)
        message = "El modelo no pudo generar una respuesta."
        yield sse("error", error={"type": "model_error", "code": "model_error", "param": None, "message": message})
        failed = turn.response("failed", error=ResponseError(code="model_error", message=message))
        yield sse("response.failed", response=dump(failed))
        return

    text = "".join(parts)
    part = OutputText(text=text)
    yield sse("response.output_text.done", **ids, text=text, logprobs=[])
    yield sse("response.content_part.done", **ids, part=dump(part))
    yield sse("response.output_item.done", output_index=0, item=dump(OutputMessage(id=turn.message_id, content=[part])))
    yield sse("response.completed", response=dump(turn.response("completed", text=text, completed=completed)))
    await _remember(store, turn, messages, text, completed, started)
