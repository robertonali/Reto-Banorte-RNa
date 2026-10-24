# Decisiones técnicas

El criterio del reto es *proporcionalidad*: cada pieza existe porque resuelve un problema concreto
del agente, y donde se pudo, la decisión se tomó con datos. Cada sección sigue el mismo orden:
contexto, opciones, decisión y evidencia.

---

## 1. Interfaz: endpoint Open Responses, sin frontend

**Contexto.** La plataforma del reto tiene su propio chat y llama a un endpoint compatible con
Open Responses. El agente Guía no especificó si la plataforma usa streaming, si manda el historial
completo o un `previous_response_id`, ni el timeout.

**Decisión.**
- FastAPI con `POST /v1/responses` que **soporta todas las variantes**: con y sin streaming,
  historial completo y `previous_response_id`.
- Respuestas con las 31 claves que exige el schema y errores con el formato de la spec (incluido 400 en lugar del 422 de FastAPI).
- Política de "flexible al recibir, estricto al enviar": se aceptan campos e items que no se usan
  (reasoning, imágenes, parámetros de muestreo) y se ignoran sin fallar.
- **Sin frontend**: la plataforma es la interfaz. Un formulario para subir CVs sería superficie de
  ataque (cualquiera podría reemplazar el perfil o meterle instrucciones) sin aportar nada al reto.

**Evidencia.** Los tests verifican las claves, el orden de los eventos SSE (`response.created` →
`in_progress` → `output_item.added` → `content_part.added` → deltas → `output_text.done` →
`content_part.done` → `output_item.done` → `response.completed`), los números de secuencia, que los
deltas sumen el texto final, y el par `error` + `response.failed` cuando falla el modelo. Además, cada
request registra su *forma* (nunca su contenido) para confirmar el contrato real de la plataforma.

**Contrato confirmado en producción.** Los logs de Render, con el agente ya registrado en la plataforma, mostraron:

| Campo | Valor observado | Consecuencia |
|---|---|---|
| `stream` | `true` en todos los requests | La ruta SSE es la que usa la plataforma. |
| `input` | lista con **todo el historial** (1, 3, 5… 15 items) | Hace falta recortar el historial, no rechazarlo (ver sección 6). |
| `previous_response_id` | nunca se envía | Postgres no es necesario para esta plataforma (ver sección 8). |
| `instructions` | nunca se envía | Solo aplican las reglas del agente. |
| Autenticación | `Authorization: Bearer` | Se quitó la alternativa `x-api-key`. |
| Cliente | `Bun/1.3.14` | — |

En producción respondió 8/8 requests con 200, con latencia de 0.6–2 s, 4,597 tokens de caché en cada
request y 0.0001–0.0004 USD por respuesta. Es un buen ejemplo de para qué sirve la observabilidad:
el diseño soportaba todas las variantes, y los datos reales dijeron qué partes importaban.

## 2. Contexto: perfil completo en el prompt (RAG descartado con datos)

**Contexto.** El CV tiene unas 700 palabras; con el detalle de proyectos, el perfil completo ocupa
~3,500 tokens.

**Opciones evaluadas.**

| Estrategia | Cómo funciona |
|---|---|
| `completo` | Todo el perfil en el system prompt, cacheado. |
| `herramientas` | Solo un índice en el prompt. El modelo pide secciones (`consultar_perfil`) o el detalle de un proyecto (`detalle_proyecto`). |
| RAG (pgvector) | Embeddings por sección y búsqueda semántica. |

**Decisión.** `completo` en producción; `herramientas` implementada y evaluada; RAG no se implementó.

**Por qué no RAG.** Con 3.5k tokens, todo el corpus cabe en el contexto con margen de sobra. RAG
agregaría latencia (embedding + búsqueda), otro servicio que operar y un modo de falla nuevo
(recuperar el fragmento equivocado), y el beneficio sería nulo: no hay nada que filtrar. Sería la decisión
correcta si el perfil creciera a decenas de documentos (publicaciones, documentación de proyectos). La
herramienta `detalle_proyecto` es justo el punto donde entraría.

**Evidencia.** Ver la comparación `completo` vs `herramientas` en la sección 5.

## 3. Un solo agente, sin A2A, multiagente ni MCP

**Decisión.** Un agente con dos herramientas de solo lectura.

**Por qué.**
- **Multiagente / A2A:** separar "agente de experiencia", "agente de proyectos" y "validador" agrega
  latencia y costo por cada salto, abre la puerta a contradicciones entre agentes y complica evaluar y
  operar, todo para un perfil de 3.5k tokens que un solo modelo lee completo.
- **MCP:** serviría para exponer el perfil como herramienta a *otros* clientes (por ejemplo, Claude
  Desktop). La plataforma del reto habla Open Responses, no MCP, así que hoy no tiene consumidor.

## 4. Proveedores detrás de una interfaz común

**Decisión.** `llm/base.py` define mensajes, herramientas, uso de tokens y eventos (`TextDelta`,
`ToolCalled`, `Completed`). Cada proveedor implementa **su propio loop de herramientas** con el
formato nativo de su API.

**Por qué no un formato neutral para todo el loop.** Los dos proveedores guardan estado distinto
entre rondas: Claude devuelve bloques de *thinking* que deben reenviarse sin cambios (historial
append-only), y OpenAI devuelve *reasoning items* cifrados (`store=false` +
`include=["reasoning.encrypted_content"]`, así ninguna conversación queda guardada en OpenAI).
Traducirlos a un formato común los rompería. El agente solo ve eventos, y cambiar de modelo es
cambiar `LLM_MODEL`.

**Detalles por proveedor.**
- **Claude:** prompt caching del bloque estable, `effort: low` (chat, no razonamiento profundo) y
  fallback del lado del servidor (`fallbacks: "default"`) en Opus/Sonnet 5.5 por si un clasificador
  de seguridad rechaza la petición.
- **OpenAI:** `prompt_cache_key` y `reasoning.effort: low`, con herramientas en modo estricto.

## 5. Elección de modelo con evals

**Contexto.** El agente debe ser correcto y fiel al perfil, y además rápido (es un chat) y barato
(los créditos también pagan el tráfico de los evaluadores).

**Método.** Seis modelos en tres niveles de precio, emparejados entre proveedores:

| Nivel | Claude | GPT | USD por millón de tokens (entrada / salida) |
|---|---|---|---|
| Económico | Haiku 5.5 | GPT-6 Luna | 0.10 / 0.50 |
| Medio | Sonnet 5.5 | GPT-6.1 Sol | 2 / 10 |
| Alto | Opus 5.5 | GPT-5.6 Sol | 4 / 20 |

Todos corren el mismo golden set de 35 casos, con el mismo prompt, `effort: low` y el mismo panel de jueces.

**Iteración v1 → v2 (el hallazgo más útil).**

| Modelo | Fidelidad v1 → v2 | Alucinación v1 → v2 | Palabras v1 → v2 |
|---|---|---|---|
| claude-opus-5-5 | 93% → 98% | 21% → 6% | 86 → 75 |
| claude-sonnet-5-5 | 95% → 99% | 14% → 3% | 88 → 75 |
| gpt-6.1-sol | 96% → 100% | 9% → 0% | 44 → 39 |
| gpt-5.6-sol | 97% → 99% | 9% → 3% | 40 → 29 |
| claude-haiku-5-5 | 99% → 99% | 3% → 3% | 65 → 61 |
| gpt-6-luna | 99% → 98% | 3% → 4% | 29 → 29 |

- En la v1, los modelos **más grandes alucinaban más**. No inventaban datos duros: "adornaban". Por
  ejemplo, "su equipo quedó en el top 8" (el CV no menciona un equipo), "Python es su lenguaje
  principal" o "usó Embedded C en Continental" (el CV lo lista como habilidad, sin ligarlo a ese empleo).
- La v2 agregó una regla explícita contra esos detalles plausibles y un límite de largo, y bajó la tasa
  de alucinación en todos los modelos grandes.
- La v2 también corrigió un falso positivo del juez (no sabía que el agente recibe la fecha de hoy), así
  que parte de la mejora viene del juez. La comparación es indicativa, no un experimento controlado.

**Decisión.**
- **Producción: `claude-haiku-5-5`.** 100% de correctitud y de comportamiento, 99% de fidelidad, el
  primer token más rápido (0.75 s) y ~0.0002 USD por respuesta: con 1 USD alcanzan ~5,000 respuestas.
- **Respaldo automático: `gpt-6.1-sol`**, de otro proveedor. Fue el único sin casos marcados, pero es
  2× más lento y 6× más caro. Si Anthropic falla antes de emitir texto, el agente responde con GPT
  (probado en tests y con una falla real), y la respuesta reporta qué modelo contestó.
- **Los modelos grandes no se justifican aquí.** Opus 5.5 cuesta ~45× más que Haiku, tarda el doble y
  no fue más correcto. La tarea (leer 3.5k tokens y contestar con precisión) no necesita más capacidad.
- **La estrategia `herramientas` queda descartada.** Bajó la correctitud (96–97%), duplicó el tiempo al
  primer token (cada consulta es una ronda extra al modelo) y una vez respondió en español a una
  pregunta en inglés. Con un perfil que cabe completo en el contexto, las herramientas solo agregan pasos.

**Honestidad estadística.** Con 35 casos, una diferencia de 1–3 puntos equivale a uno o dos casos y
está dentro del ruido (además, los jueces LLM tienen su propia varianza). Por eso, entre modelos
empatados en calidad, la decisión la tomaron la latencia y el costo, que sí difieren por órdenes de
magnitud. Cambiar de modelo es cambiar `LLM_MODEL`.

**Costo de evaluar.** Las dos corridas completas (14 configuraciones × 35 casos, con agente y dos
jueces) costaron ~3.6 USD. Los jueces costaron más que los agentes evaluados.

## 6. Grounding y guardrails proporcionales al riesgo

| Riesgo | Mitigación |
|---|---|
| Inventar experiencia | Reglas explícitas ("solo el perfil", "no agregues detalles plausibles"). Se mide con fidelidad y la tasa de alucinación. |
| Datos personales | **Privacidad por construcción:** el agente solo ve el perfil, y el perfil solo contiene lo que Roberto decidió compartir. No hace falta filtrar la salida. |
| Prompt injection | Reglas de seguridad en el prompt, más detección heurística (`guardrails.py`) que **no bloquea** (habría falsos positivos), pero agrega un recordatorio en la parte dinámica del prompt y lo registra en el log. Los evals tienen 3 casos de injection. |
| Instrucciones del cliente | `instructions` y los mensajes system/developer van a la parte dinámica como "preferencias de tono y formato", con menor prioridad que las reglas. |
| Abuso / costo | API key obligatoria en producción, rate limit por IP, límite de caracteres por mensaje del usuario, máximo 4 rondas de herramientas y timeouts. |
| Conversaciones largas | La plataforma reenvía todo el historial, así que cada turno suma 2 mensajes. Con un límite que rechazara, el chat se rompería en la pregunta 21. El historial se **recorta** a los 40 mensajes más recientes (empezando por uno del usuario), así el chat nunca falla y el costo por request queda acotado. |
| Temas delicados | Salario, motivos de cambio y disponibilidad: el agente redirige a Roberto. |

## 7. Evals: golden set, panel de jueces y MLflow

- **35 casos en 8 categorías:** dato correcto, inferencia, no disponible, trampa de alucinación,
  fuera de alcance, injection, multi-turno y ambigua. Cada caso define el *comportamiento esperado*
  (responder, no_disponible, redirigir, rechazar) y una respuesta de referencia.
- **Dos capas de calificación:**
  1. Checks deterministas (gratis y reproducibles): términos obligatorios y prohibidos, sin importar
     acentos ni mayúsculas, con coincidencia de palabra completa.
  2. LLM como juez con rúbrica fija, en **panel de dos familias** (Claude Sonnet 5.5 + GPT-6.1 Sol)
     cuyas calificaciones se promedian, para reducir el sesgo de autopreferencia.
- **MLflow** registra por corrida los parámetros, las métricas (correctitud, fidelidad, tasa de
  alucinación, comportamiento por categoría, concisión, latencia p50/p95, primer token, costo) y la
  tabla de casos con las justificaciones de los jueces.
- **También se auditó al juez.** En la v1, los jueces castigaban los cálculos de antigüedad porque no
  sabían que el agente recibe la fecha de hoy. Se corrigió la rúbrica en la v2.

## 8. Estado de conversación y degradación elegante

**En producción, sin Postgres.** Los logs mostraron que la plataforma manda el historial completo y
nunca usa `previous_response_id` (sección 1), así que el despliegue no usa la base: una pieza menos
que operar y pagar. El soporte queda en el código para clientes que sí lo usen.

`previous_response_id` exige recuperar la conversación anterior, así que hace falta guardarla. Opciones:
- **Memoria:** se pierde en cada deploy o reinicio.
- **Postgres:** es lo que se usa. Es un servicio administrado en el mismo Render, y la misma tabla
  guarda métricas de operación (modelo, tokens, costo, latencia), así que la base sirve también como registro.

Si Postgres falla al arrancar, el servicio **sigue respondiendo** con conversaciones en memoria; solo
se pierde la continuidad tras reinicios.

Las consultas son síncronas (psycopg) y corren en un hilo con `asyncio.to_thread`: funciona igual en
Linux y Windows, y la carga de un agente de CV no requiere más.

## 9. Despliegue y operación

- **Docker + Render Blueprint (`render.yaml`):** la infraestructura queda versionada, con web service
  y Postgres declarados. La imagen corre sin root, con dependencias exactas de `uv.lock` y sin
  MLflow ni herramientas de desarrollo.
- **Instancia que no se duerme:** en el plan free, el primer mensaje del evaluador tardaría cerca de
  un minuto o fallaría por timeout.
- **`autoDeployTrigger: checksPass`:** Render solo despliega si los tests de GitHub Actions pasan.
- **El perfil es un Secret File** (`/etc/secrets/profile.yaml`): no está en git ni en la imagen.
- **Configuración que falla cerrado:** `ENVIRONMENT` vale `production` por defecto, y en producción
  el servicio no arranca sin `AGENT_API_KEY`. Un olvido no puede dejar el endpoint abierto.
- **Observabilidad:** logs por request con su id, la forma de cada request (sin contenido) y una línea
  por respuesta con modelo, tokens, caché, costo y latencia.

## 10. Datos personales fuera del repo

El repo es público y el CV no.
- **`private/`** (ignorado por git) contiene el CV, el perfil, el golden set real y los resultados de los evals.
- **Hook de pre-commit y GitHub Actions** corren `scripts/check_sensitive.py`: bloquea archivos
  privados y cualquier dato de contacto del perfil (los teléfonos se buscan por dígitos, en cualquier formato).
- **`.dockerignore`** deja fuera `private/`, así que la imagen no puede contener el perfil.
- **El repo usa ejemplos ficticios** (`data/profile.example.yaml`, `evals/golden_set.example.yaml`)
  para que cualquiera pueda correr los tests y los evals.

## 11. Ingesta del CV: borrador automático + revisión humana

`scripts/extract_profile.py` convierte un CV (`.tex`, `.md`, `.txt` o `.pdf`) en un **borrador** de
perfil con *structured outputs*. El borrador nunca reemplaza al perfil revisado.

**Por qué la revisión humana.** El perfil es la fuente de verdad: un error de extracción que nadie
revisa se vuelve una "alucinación" que el agente repetiría con total seguridad.

**Evidencia.** El borrador generado desde el LaTeX coincidió en estructura con el perfil revisado a
mano: 3 empleos, F1Tenth como proyecto, educación con fechas e idiomas vacíos (el CV no los menciona).

**Lección técnica.** Con campos opcionales y diccionarios libres, el esquema generaba una gramática
demasiado grande para *structured outputs* y la API lo rechazaba. Se simplificó a solo textos y
listas (texto vacío = dato ausente), y la conversión al modelo final se hace en el código.
