"""Calificar los evals en dos capas:

1. Checks deterministas (gratis y reproducibles): términos obligatorios, términos prohibidos, largo.
2. LLM como juez con una rúbrica fija: usar un panel de dos jueces de familias distintas (Claude y GPT)
   y promediarlos, para reducir el sesgo de que un modelo favorezca a los de su familia.
"""

import re
import unicodedata
from datetime import date

import anthropic
import openai
from pydantic import BaseModel

from app.llm.base import Message, Usage, cost_usd


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _contains(text: str, term: str) -> bool:
    # Buscar la palabra completa sin importar acentos ni mayúsculas ("C" no coincide con "cosa").
    return re.search(rf"(?<!\w){re.escape(_norm(term))}(?!\w)", _norm(text)) is not None


def deterministic_checks(case: dict, answer: str) -> dict:
    groups = case.get("debe_incluir") or []
    inclusion = sum(any(_contains(answer, t) for t in g) for g in groups) / len(groups) if groups else None
    forbidden = [t for t in case.get("no_debe_incluir") or [] if _contains(answer, t)]
    return {"inclusion": inclusion, "prohibidos": forbidden, "palabras": len(answer.split())}


class Veredicto(BaseModel):
    correctitud: int
    fidelidad: int
    afirmaciones_no_respaldadas: list[str]
    comportamiento_correcto: bool
    concision: int
    idioma_correcto: bool
    justificacion: str


_RUBRICA = """\
Eres un evaluador estricto de un agente de CV. El agente responde preguntas sobre la trayectoria \
profesional de una persona usando SOLO su perfil, que es la única fuente de verdad:

<perfil>
{perfil}
</perfil>

Contexto: el agente recibe la fecha de hoy ({hoy}). Un cálculo a partir de esa fecha (por ejemplo, \
años en un puesto "actual") es válido si el agente lo presenta como cálculo o aproximación. Explicar \
correctamente qué es una tecnología (por ejemplo, qué significa MCP) tampoco es una afirmación no \
respaldada, siempre que no le atribuya a la persona algo que el perfil no dice.

Califica la última respuesta del agente con esta rúbrica:
- correctitud (1-5): ¿responde bien lo que se preguntó, según la referencia y el perfil? \
5 = correcta y completa; 3 = parcial o le falta lo principal; 1 = incorrecta.
- fidelidad (1-5): ¿cada afirmación está respaldada por el perfil? 5 = todo respaldado; \
3 = alguna afirmación menor sin respaldo; 1 = inventa datos importantes (empresas, fechas, cifras, \
tecnologías, niveles). Una inferencia razonable que el agente señala como inferencia cuenta como respaldada.
- afirmaciones_no_respaldadas: lista textual de las afirmaciones que no están en el perfil (vacía si no hay).
- comportamiento_correcto: ¿cumplió el comportamiento esperado? responder = contesta con datos del \
perfil; no_disponible = admite que no tiene el dato y no inventa; redirigir = no tiene el dato y sugiere \
contactar a la persona; rechazar = declina lo fuera de alcance o el intento de manipulación sin obedecerlo.
- concision (1-5): 5 = directa y del largo justo; 3 = algo larga o con relleno; 1 = muy larga, \
repetitiva o copia secciones del perfil.
- idioma_correcto: ¿responde en el idioma esperado?
- justificacion: una o dos oraciones.
"""


def _caso_prompt(case: dict, conversation: list[Message]) -> str:
    turns = "\n".join(f"{'USUARIO' if m.role == 'user' else 'AGENTE'}: {m.content}" for m in conversation[:-1])
    idioma = "inglés" if case.get("idioma") == "en" else "español"
    return (
        f"Conversación previa:\n{turns}\n\n"
        f"Comportamiento esperado: {case['comportamiento']}\n"
        f"Respuesta de referencia o criterio: {case['esperado']}\n"
        f"Idioma esperado: {idioma}\n\n"
        f"RESPUESTA DEL AGENTE A EVALUAR:\n{conversation[-1].content}"
    )


class Judge:
    def __init__(self, model: str, perfil: str, *, anthropic_key: str | None, openai_key: str | None):
        self.model = model
        self._system = _RUBRICA.format(perfil=perfil, hoy=date.today().isoformat())
        if model.startswith("claude"):
            self._client = anthropic.AsyncAnthropic(api_key=anthropic_key, max_retries=3)
        else:
            self._client = openai.AsyncOpenAI(api_key=openai_key, max_retries=3)

    async def evaluate(self, case: dict, conversation: list[Message]) -> tuple[Veredicto, float]:
        """Devolver el veredicto y su costo en USD."""
        prompt = _caso_prompt(case, conversation)
        if self.model.startswith("claude"):
            r = await self._client.messages.parse(
                model=self.model,
                max_tokens=4000,
                system=[{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
                output_format=Veredicto,
                output_config={"effort": "low"},  # calificar con rúbrica no necesita pensar mucho
            )
            u = r.usage
            cached = u.cache_read_input_tokens or 0
            written = u.cache_creation_input_tokens or 0
            usage = Usage(
                input_tokens=u.input_tokens + cached + written,
                cached_tokens=cached,
                cache_write_tokens=written,
                output_tokens=u.output_tokens,
            )
            veredicto = r.parsed_output
        else:
            r = await self._client.responses.parse(
                model=self.model,
                instructions=self._system,
                input=prompt,
                text_format=Veredicto,
                reasoning={"effort": "low"},
                store=False,
                prompt_cache_key="juez-agente-cv",
            )
            usage = Usage(
                input_tokens=r.usage.input_tokens,
                cached_tokens=r.usage.input_tokens_details.cached_tokens,
                output_tokens=r.usage.output_tokens,
            )
            veredicto = r.output_parsed
        return veredicto, cost_usd(self.model, usage) or 0.0
