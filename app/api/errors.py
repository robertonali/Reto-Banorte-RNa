"""Responder errores con el formato de Open Responses: {"error": {"type", "code", "param", "message"}}."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agent.guardrails import GuardrailError

logger = logging.getLogger("cv_agent")

# Tipos de error de la spec (tabla "error types").
_TYPES = {400: "invalid_request", 401: "invalid_request", 404: "not_found", 429: "too_many_requests"}
_CODES = {401: "invalid_api_key", 404: "not_found", 429: "rate_limit_exceeded"}


class OpenResponsesError(Exception):
    def __init__(self, status: int, message: str, *, type: str, code: str, param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type, code, param


def error_response(
    status: int, message: str, *, type: str, code: str | None, param: str | None = None, headers=None
) -> JSONResponse:
    body = {"error": {"type": type, "code": code, "param": param, "message": message}}
    return JSONResponse(status_code=status, content=body, headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(OpenResponsesError)
    async def _own(request: Request, exc: OpenResponsesError) -> JSONResponse:
        return error_response(exc.status, exc.message, type=exc.type, code=exc.code, param=exc.param)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error_type = _TYPES.get(exc.status_code, "server_error" if exc.status_code >= 500 else "invalid_request")
        return error_response(
            exc.status_code, str(exc.detail), type=error_type, code=_CODES.get(exc.status_code), headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Responder 400 invalid_request como pide la spec (FastAPI usa 422 por defecto).
        # En campos con varias formas (texto o lista), reportar el error más profundo: es el más útil.
        error = max(exc.errors(), key=lambda e: len(e["loc"]))
        param = ".".join(str(x) for x in error["loc"] if x != "body") or None
        return error_response(400, error["msg"], type="invalid_request", code="invalid_value", param=param)

    @app.exception_handler(GuardrailError)
    async def _guardrail(request: Request, exc: GuardrailError) -> JSONResponse:
        return error_response(400, str(exc), type="invalid_request", code="invalid_input", param=exc.param)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Error inesperado en %s %s", request.method, request.url.path)
        return error_response(500, "Error interno del servidor.", type="server_error", code="server_error")
