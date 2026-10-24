"""Rate limit en memoria: N requests por minuto por cliente (IP).

Proteger mis créditos de OpenAI/Anthropic si alguien abusa del endpoint. En memoria basta
para una sola instancia de Render.

TODO: pasar el conteo a Redis si algún día corren varias instancias.
"""

import time
from collections import defaultdict, deque
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings

_WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > _WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True


_limiter = RateLimiter()


def rate_limit(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> None:
    client = request.client.host if request.client else "desconocido"
    if not _limiter.allow(client, settings.rate_limit_per_minute):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Demasiadas solicitudes: máximo {settings.rate_limit_per_minute} por minuto.",
            headers={"Retry-After": str(int(_WINDOW_SECONDS))},
        )
