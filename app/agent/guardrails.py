"""Guardrails proporcionales al riesgo de un agente de CV.

- Entrada: recortar el historial a los mensajes más recientes, limitar el tamaño de los mensajes
  del usuario (costo y abuso) y exigir una conversación válida.
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


def trim_history(messages: list[Message], max_messages: int) -> list[Message]:
    """Conservar solo los mensajes más recientes, empezando siempre por uno del usuario.

    La plataforma reenvía todo el historial en cada turno: recortar en lugar de rechazar, para que
    una conversación larga no se rompa y el costo por request siga acotado.
    """
    trimmed = messages[-max_messages:]
    # Quitar mensajes iniciales del asistente: Claude exige que la conversación empiece con el usuario.
    while trimmed and trimmed[0].role == "assistant":
        trimmed = trimmed[1:]
    return trimmed


def validate_conversation(messages: list[Message], *, max_message_chars: int) -> None:
    if not messages or messages[-1].role != "user":
        raise GuardrailError("La conversación debe terminar con un mensaje del usuario.", param="input")
    if not messages[-1].content.strip():
        raise GuardrailError("El mensaje del usuario está vacío.", param="input")
    # Limitar solo los mensajes del usuario: las respuestas del asistente salen del propio agente.
    for m in messages:
        if m.role == "user" and len(m.content) > max_message_chars:
            raise GuardrailError(f"Un mensaje excede el máximo de {max_message_chars} caracteres.", param="input")


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
