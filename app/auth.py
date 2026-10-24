"""Validar la API key que la plataforma del reto envía a mi endpoint."""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Exigir la key en `Authorization: Bearer <key>` o, como alternativa, en `x-api-key`.

    Sin key configurada (solo permitido en local), desactivar la autenticación.

    TODO: confirmar en los logs de Render qué encabezado usa la plataforma y dejar solo ese.
    """
    if settings.agent_api_key is None:
        return

    provided = credentials.credentials if credentials else request.headers.get("x-api-key")
    expected = settings.agent_api_key.get_secret_value()
    if not provided or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida o ausente.",
            headers={"WWW-Authenticate": "Bearer"},
        )
