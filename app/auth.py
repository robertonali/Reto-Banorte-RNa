"""Validar la API key que la plataforma del reto envía a mi endpoint."""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Exigir la key en `Authorization: Bearer <key>`, el encabezado que usa la plataforma
    (confirmado en los logs de Render).

    Sin key configurada (solo permitido en local), desactivar la autenticación.
    """
    if settings.agent_api_key is None:
        return

    provided = credentials.credentials if credentials else None
    expected = settings.agent_api_key.get_secret_value()
    if not provided or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida o ausente.",
            headers={"WWW-Authenticate": "Bearer"},
        )
