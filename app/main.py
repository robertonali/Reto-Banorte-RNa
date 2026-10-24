"""app de FastAPI"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api import responses
from app.api.errors import register_error_handlers
from app.config import get_settings
from app.profile import load_profile
from app.store import MemoryStore, PostgresStore

logger = logging.getLogger("cv_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()  # validar la configuración: si es inválida, el servicio no arranca
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # silenciar la línea por llamada al LLM: es ruido
    if settings.agent_api_key is None:
        logger.warning("AGENT_API_KEY no está configurada: autenticación desactivada (solo local).")

    # Sin perfil no hay agente: mejor fallar al arrancar que con el primer usuario.
    app.state.profile = load_profile(settings.profile_path)
    app.state.agent = None  # construirlo en el primer request (ver api/responses.get_agent)

    app.state.store = MemoryStore()
    if settings.database_url:
        store = PostgresStore(settings.database_url.get_secret_value())
        try:
            await store.open()
            app.state.store = store
        except Exception:
            # Seguir respondiendo sin base; solo se pierde previous_response_id tras reinicios.
            logger.exception("No se pudo conectar a Postgres: las conversaciones se guardan en memoria.")
    else:
        logger.warning("DATABASE_URL no está configurada: las conversaciones se guardan en memoria.")

    logger.info(
        "Servicio iniciado (environment=%s, modelo=%s, estrategia=%s, perfil=%s)",
        settings.environment,
        settings.llm_model,
        settings.context_strategy,
        app.state.profile.nombre,
    )
    yield
    if isinstance(app.state.store, PostgresStore):
        await app.state.store.close()


app = FastAPI(title="Agente de CV — Reto IA Banorte", version="0.1.0", lifespan=lifespan)
register_error_handlers(app)
app.include_router(responses.router)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Registrar una línea de log por request con su id, estado y duración (en streaming: hasta el primer byte)."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s -> %s (%.0f ms) request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
        request_id,
    )
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check para Render; no requiere autenticación."""
    return {"status": "ok"}
