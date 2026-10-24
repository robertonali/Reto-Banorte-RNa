"""Contrato HTTP: auth, formato Open Responses, streaming, continuación y errores."""

import json

from app import ratelimit
from app.config import Settings, get_settings
from app.main import app
from tests.conftest import AUTH, FakeProvider

# Claves que la spec exige en toda respuesta (ResponseResource.required).
REQUIRED_KEYS = set(
    "id object created_at completed_at status incomplete_details model previous_response_id instructions "
    "output error tools tool_choice truncation parallel_tool_calls text top_p presence_penalty frequency_penalty "
    "top_logprobs temperature reasoning usage max_output_tokens max_tool_calls store background service_tier "
    "metadata safety_identifier prompt_cache_key".split()
)


def parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        data = json.loads(fields["data"])
        assert fields["event"] == data["type"]  # la spec exige que coincidan
        events.append(data)
    return events


def test_health_no_requiere_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_sin_key_devuelve_401_con_formato_de_la_spec(client):
    r = client.post("/v1/responses", json={"input": "hola"})
    assert r.status_code == 401
    assert r.json()["error"] == {
        "type": "invalid_request",
        "code": "invalid_api_key",
        "param": None,
        "message": "API key inválida o ausente.",
    }


def test_solo_acepta_bearer(client):
    # La plataforma usa Authorization: Bearer (confirmado en los logs de Render).
    assert client.post("/v1/responses", json={"input": "hola"}, headers=AUTH).status_code == 200
    assert client.post("/v1/responses", json={"input": "hola"}, headers={"Authorization": "Bearer otra"}).status_code == 401
    assert client.post("/v1/responses", json={"input": "hola"}, headers={"x-api-key": "clave-de-prueba"}).status_code == 401


def test_respuesta_sin_streaming_cumple_la_spec(client):
    r = client.post("/v1/responses", json={"input": "¿Qué estudió?", "metadata": {"origen": "test"}}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= REQUIRED_KEYS
    assert body["object"] == "response" and body["status"] == "completed"
    assert body["id"].startswith("resp_") and body["output"][0]["id"].startswith("msg_")
    assert body["output"][0]["content"][0] == {"type": "output_text", "text": "Hola, soy el asistente.", "annotations": []}
    assert body["usage"]["total_tokens"] == 110
    assert body["usage"]["input_tokens_details"]["cached_tokens"] == 80
    assert body["metadata"] == {"origen": "test"}


def test_streaming_emite_la_secuencia_oficial(client):
    r = client.post("/v1/responses", json={"input": "hola", "stream": True}, headers=AUTH)
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    types = [e["type"] for e in events]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    done = next(e for e in events if e["type"] == "response.output_text.done")
    assert deltas == done["text"] == "Hola, soy el asistente."
    final = events[-1]["response"]
    assert final["status"] == "completed" and set(final) >= REQUIRED_KEYS


def test_previous_response_id_carga_la_conversacion(client, fake):
    first = client.post("/v1/responses", json={"input": "Primera pregunta"}, headers=AUTH).json()
    r = client.post(
        "/v1/responses", json={"input": "Segunda pregunta", "previous_response_id": first["id"]}, headers=AUTH
    )
    assert r.status_code == 200 and r.json()["previous_response_id"] == first["id"]
    sent = [(m.role, m.content) for m in fake.calls[-1]["messages"]]
    assert sent == [
        ("user", "Primera pregunta"),
        ("assistant", "Hola, soy el asistente."),
        ("user", "Segunda pregunta"),
    ]


def test_previous_response_id_inexistente_da_404(client):
    r = client.post("/v1/responses", json={"input": "x", "previous_response_id": "resp_nada"}, headers=AUTH)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "previous_response_not_found"
    assert r.json()["error"]["param"] == "previous_response_id"


def test_historial_completo_y_mensajes_de_sistema(client, fake):
    body = {
        "input": [
            {"type": "message", "role": "developer", "content": "Responde breve."},
            {"role": "assistant", "content": "Saludo inicial de la plataforma"},
            {"role": "user", "content": [{"type": "input_text", "text": "Hola"}]},
            {"role": "assistant", "content": "¡Hola!"},
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"role": "user", "content": "¿Dónde trabaja?"},
        ],
        "instructions": "Usa un tono formal.",
    }
    assert client.post("/v1/responses", json=body, headers=AUTH).status_code == 200
    call = fake.calls[-1]
    # El saludo inicial del asistente se descarta (Claude exige empezar con el usuario).
    assert [m.role for m in call["messages"]] == ["user", "assistant", "user"]
    # Las instrucciones del cliente van en la parte dinámica, con menor prioridad que las reglas.
    assert "Usa un tono formal." in call["system"].dynamic and "Responde breve." in call["system"].dynamic
    assert "Usa un tono formal." not in call["system"].stable


def test_body_invalido_da_400_invalid_request(client):
    r = client.post("/v1/responses", json={"input": [{"role": "robot", "content": "x"}]}, headers=AUTH)
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["type"] == "invalid_request" and "role" in error["param"]


def test_guardrails_de_entrada(client):
    largo = client.post("/v1/responses", json={"input": "x" * 5000}, headers=AUTH)
    assert largo.status_code == 400 and largo.json()["error"]["code"] == "invalid_input"
    sin_usuario = client.post("/v1/responses", json={"input": []}, headers=AUTH)
    assert sin_usuario.status_code == 400


def test_conversacion_larga_se_recorta_en_vez_de_fallar(client, fake):
    # Así llama la plataforma: historial completo en cada turno. 45 mensajes rebasan el límite de 40.
    historial = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"mensaje {i}"} for i in range(45)
    ]
    r = client.post("/v1/responses", json={"input": historial, "stream": True}, headers=AUTH)
    assert r.status_code == 200
    enviados = fake.calls[-1]["messages"]
    assert len(enviados) == 39 and enviados[0].role == "user" and enviados[-1].content == "mensaje 44"


def test_falla_del_modelo_sin_streaming(client, fake):
    fake.fail = True
    r = client.post("/v1/responses", json={"input": "hola"}, headers=AUTH)
    assert r.status_code == 500 and r.json()["error"]["type"] == "model_error"


def test_falla_del_modelo_en_streaming(client, fake):
    fake.fail = True
    events = parse_sse(client.post("/v1/responses", json={"input": "hola", "stream": True}, headers=AUTH).text)
    assert [e["type"] for e in events[-2:]] == ["error", "response.failed"]
    assert events[-1]["response"]["status"] == "failed"


def test_failover_a_otro_proveedor(client, profile):
    from app.agent.agent import Agent, AgentConfig, Fallback
    from app.api.responses import get_agent

    respaldo = FakeProvider(parts=("Respuesta ", "del respaldo."))
    agent = Agent(
        profile,
        FakeProvider(fail=True),
        AgentConfig(model="claude-haiku-5-5", strategy="completo", effort="low", max_output_tokens=500),
        fallback=Fallback(respaldo, "gpt-6.1-sol"),
    )
    app.dependency_overrides[get_agent] = lambda: agent
    body = client.post("/v1/responses", json={"input": "hola"}, headers=AUTH).json()
    assert body["output"][0]["content"][0]["text"] == "Respuesta del respaldo."
    assert body["model"] == "gpt-6.1-sol"  # la respuesta dice qué modelo contestó en realidad


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(ratelimit, "_limiter", ratelimit.RateLimiter())
    app.dependency_overrides[get_settings] = lambda: Settings(
        environment="local", agent_api_key="clave-de-prueba", rate_limit_per_minute=2
    )
    codes = [client.post("/v1/responses", json={"input": "hola"}, headers=AUTH).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
