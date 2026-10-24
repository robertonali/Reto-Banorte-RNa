"""JSON de Open Responses (https://www.openresponses.org/specification).

Ser flexible con lo que se recibe y estricto con lo que se envía: aceptar requests con campos
que no se usan, pero responder siempre con todos los campos que exige la spec.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# ─── Request: lo que manda la plataforma ────────────────────────────────────


class ContentPart(BaseModel):
    """Una parte del contenido de un mensaje, p. ej. {"type": "input_text", "text": "Hola"}."""

    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None


class InputItem(BaseModel):
    """Un elemento del historial. Normalmente un mensaje: {"role": "user", "content": "Hola"}."""

    model_config = ConfigDict(extra="allow")

    type: str = "message"
    role: Literal["user", "assistant", "system", "developer"] | None = None
    content: str | list[ContentPart] | None = None


class ResponseRequest(BaseModel):
    """Cuerpo de POST /v1/responses. La spec no hace obligatorio ningún campo."""

    model: str | None = None
    input: str | list[InputItem] | None = None
    instructions: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    metadata: dict[str, str] | None = None


# ─── Response: lo que devuelve el agente ────────────────────────────────────


class OutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict] = []


class OutputMessage(BaseModel):
    type: Literal["message"] = "message"
    id: str
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    role: Literal["assistant"] = "assistant"
    content: list[OutputText]


class InputTokensDetails(BaseModel):
    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: InputTokensDetails = InputTokensDetails()
    output_tokens_details: OutputTokensDetails = OutputTokensDetails()


class ResponseError(BaseModel):
    code: str
    message: str


class ResponseObject(BaseModel):
    """Respuesta de POST /v1/responses. La spec exige que todas las claves existan, aunque sean null."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    completed_at: int | None = None
    status: Literal["queued", "in_progress", "completed", "failed", "incomplete"]
    incomplete_details: dict[str, str] | None = None
    model: str
    previous_response_id: str | None = None
    instructions: str | None = None
    output: list[OutputMessage] = []
    error: ResponseError | None = None
    usage: Usage | None = None
    metadata: dict[str, str] = {}

    # Parámetros de generación: la spec pide reportarlos. El cliente no los controla,
    # así que devolver valores fijos.
    tools: list[dict] = []
    tool_choice: str = "auto"
    truncation: Literal["auto", "disabled"] = "disabled"
    parallel_tool_calls: bool = False
    text: dict = {"format": {"type": "text"}}
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    top_logprobs: int = 0
    temperature: float = 1.0
    reasoning: dict | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    store: bool = True
    background: bool = False
    service_tier: str = "default"
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
