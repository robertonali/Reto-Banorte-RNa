"""Guardrails proporcionales al riesgo de un agente de CV.

- Entrada: límites de tamaño (costo y abuso) y una conversación válida.
- Prompt injection: detectar con heurísticas. No bloquear (habría falsos positivos): reforzar
  las reglas en el prompt y registrarlo en los logs.
- Privacidad: por construcción. El agente solo ve el perfil, y el perfil solo contiene lo que
  decidí compartir; no hay nada que filtrar en la salida.
"""

import re

from app.llm.base import Message


class GuardrailError(ValueError):
    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.param = param


def validate_conversation(messages: list[Message], *, max_message_chars: int, max_history_messages: int) -> None:
    if not messages or messages[-1].role != "user":
        raise GuardrailError("La conversación debe terminar con un mensaje del usuario.", param="input")
    if len(messages) > max_history_messages:
        raise GuardrailError(
            f"La conversación excede el máximo de {max_history_messages} mensajes.", param="input"
        )
    for m in messages:
        if len(m.content) > max_message_chars:
            raise GuardrailError(f"Un mensaje excede el máximo de {max_message_chars} caracteres.", param="input")
    if not messages[-1].content.strip():
        raise GuardrailError("El mensaje del usuario está vacío.", param="input")


_INJECTION_PATTERNS = {
    "ignorar_instrucciones": r"\b(ignora|olvida|omite|ignore|forget|disregard)\b.{0,40}\b(instrucciones|reglas|indicaciones|instructions|rules|prompt)\b",
    "revelar_prompt": r"\b(system prompt|prompt del sistema|tus instrucciones|your instructions|reveal|revela)\b",
    "cambio_de_rol": r"\b(ahora eres|act[uú]a como|finge|pretend|you are now|developer mode|modo desarrollador|jailbreak)\b",
    "afirmacion_forzada": r"\b(di|afirma|escribe|responde)\s+que\s+([eé]l|ella|tiene|es|sabe|eres)\b|\b(say|claim)\s+that\b",
}
_COMPILED = {name: re.compile(p, re.IGNORECASE) for name, p in _INJECTION_PATTERNS.items()}


def injection_signals(text: str) -> list[str]:
    """Devolver los nombres de los patrones de injection presentes en el texto (vacío si no hay)."""
    return [name for name, pattern in _COMPILED.items() if pattern.search(text)]
