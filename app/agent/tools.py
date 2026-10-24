"""Herramientas del agente (estrategia "herramientas"): consultar el perfil por partes.

Solo leen el perfil y no tocan nada más, así que un modelo manipulado no puede usarlas
para hacer daño.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.prompts import render_project_detail
from app.llm.base import Tool
from app.profile import (
    Profile,
    render_contacto,
    render_educacion,
    render_experiencia,
    render_habilidades,
    render_lista,
    render_proyecto,
)

Seccion = Literal["experiencia", "educacion", "habilidades", "proyectos", "logros", "cursos", "idiomas", "contacto"]


class ConsultarPerfilArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seccion: Seccion = Field(description="Sección del perfil a consultar.")


class DetalleProyectoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proyecto_id: str = Field(description="id del proyecto, tal como aparece en el índice.")


def build_tools(profile: Profile) -> list[Tool]:
    def consultar_perfil(args: ConsultarPerfilArgs) -> str:
        p = profile
        secciones = {
            "experiencia": lambda: render_experiencia(p),
            "educacion": lambda: render_educacion(p),
            "habilidades": lambda: render_habilidades(p),
            "proyectos": lambda: "\n\n".join(render_proyecto(pr, con_detalle=False) for pr in p.proyectos),
            "logros": lambda: render_lista(p.logros, "(sin logros registrados)"),
            "cursos": lambda: render_lista(p.cursos, "(sin cursos registrados)"),
            "idiomas": lambda: render_lista(p.idiomas, "(no hay información de idiomas)"),
            "contacto": lambda: render_contacto(p),
        }
        return secciones[args.seccion]()

    def detalle_proyecto(args: DetalleProyectoArgs) -> str:
        detalle = render_project_detail(profile, args.proyecto_id)
        if detalle is None:
            ids = ", ".join(pr.id for pr in profile.proyectos)
            return f"No existe un proyecto con id '{args.proyecto_id}'. Ids válidos: {ids}."
        return detalle

    return [
        Tool(
            name="consultar_perfil",
            description="Devuelve una sección completa del perfil profesional (experiencia, educación, "
            "habilidades, proyectos, logros, cursos, idiomas o contacto).",
            args=ConsultarPerfilArgs,
            run=consultar_perfil,
        ),
        Tool(
            name="detalle_proyecto",
            description="Devuelve el detalle completo de un proyecto: problema, arquitectura, "
            "decisiones, resultados y limitaciones.",
            args=DetalleProyectoArgs,
            run=detalle_proyecto,
        ),
    ]
