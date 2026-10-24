"""Guardar conversaciones para soportar `previous_response_id`.

- MemoryStore: para local y tests. Se pierde al reiniciar.
- PostgresStore: para producción (Render). Guardar también las métricas de cada respuesta
  (modelo, tokens, costo, latencia), para usar la base además como registro de operación.

Correr las consultas a Postgres (síncronas, con psycopg) en un hilo con asyncio.to_thread:
funciona igual en Linux y en Windows, y la carga de un agente de CV no necesita más.
"""

import asyncio
import json
from collections import OrderedDict
from dataclasses import asdict
from typing import Protocol

from psycopg_pool import ConnectionPool

from app.llm.base import Message


class ConversationStore(Protocol):
    async def get(self, response_id: str) -> list[Message] | None: ...
    async def save(self, response_id: str, messages: list[Message], meta: dict) -> None: ...


class MemoryStore:
    def __init__(self, max_items: int = 1000):
        self._items: OrderedDict[str, list[Message]] = OrderedDict()
        self._max_items = max_items

    async def get(self, response_id: str) -> list[Message] | None:
        return self._items.get(response_id)

    async def save(self, response_id: str, messages: list[Message], meta: dict) -> None:
        self._items[response_id] = messages
        if len(self._items) > self._max_items:
            self._items.popitem(last=False)  # descartar la conversación más antigua


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS responses (
    id          TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    messages    JSONB NOT NULL,
    meta        JSONB NOT NULL
)
"""


class PostgresStore:
    def __init__(self, dsn: str):
        self._pool = ConnectionPool(dsn, min_size=1, max_size=4, open=False)

    async def open(self) -> None:
        await asyncio.to_thread(self._open)

    def _open(self) -> None:
        self._pool.open(wait=True, timeout=15)
        with self._pool.connection() as conn:
            conn.execute(_CREATE_TABLE)

    async def close(self) -> None:
        await asyncio.to_thread(self._pool.close)

    async def get(self, response_id: str) -> list[Message] | None:
        return await asyncio.to_thread(self._get, response_id)

    def _get(self, response_id: str) -> list[Message] | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT messages FROM responses WHERE id = %s", (response_id,)).fetchone()
        return [Message(**m) for m in row[0]] if row else None

    async def save(self, response_id: str, messages: list[Message], meta: dict) -> None:
        await asyncio.to_thread(self._save, response_id, messages, meta)

    def _save(self, response_id: str, messages: list[Message], meta: dict) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO responses (id, messages, meta) VALUES (%s, %s, %s)",
                (response_id, json.dumps([asdict(m) for m in messages]), json.dumps(meta)),
            )
