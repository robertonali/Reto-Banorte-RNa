"""Perfil profesional: el modelo de datos, su carga desde YAML y su versión en texto para el prompt.

El perfil es la única fuente de verdad del agente. Vive fuera del repo (private/ en local,
Secret File en Render); en el repo solo hay un ejemplo ficticio (data/profile.example.yaml).
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    # Un campo mal escrito en el YAML debe fallar, no ignorarse en silencio.
    model_config = ConfigDict(extra="forbid")


class Contacto(_Model):
    email: str | None = None
    telefono: str | None = None
    linkedin: str | None = None
    github: str | None = None
    youtube: str | None = None


class Experiencia(_Model):
    id: str
    puesto: str
    empresa: str
    inicio: str
    fin: str | None = None  # None = puesto actual
    modalidad: str | None = None
    descripcion: str
    tecnologias: list[str] = []


class Educacion(_Model):
    grado: str
    institucion: str
    inicio: str | None = None
    fin: str | None = None  # None = en curso
    notas: str | None = None


class Proyecto(_Model):
    id: str
    nombre: str
    contexto: str | None = None
    fecha: str | None = None
    resumen: str
    detalle: str | None = None
    tecnologias: list[str] = []
    enlaces: list[str] = []


class Profile(_Model):
    nombre: str
    titular: str
    ubicacion: str | None = None
    resumen: str
    contacto: Contacto = Contacto()
    experiencia: list[Experiencia] = []
    educacion: list[Educacion] = []
    habilidades: dict[str, list[str]] = {}
    proyectos: list[Proyecto] = []
    logros: list[str] = []
    cursos: list[str] = []
    idiomas: list[str] = []

    def proyecto(self, proyecto_id: str) -> Proyecto | None:
        return next((p for p in self.proyectos if p.id == proyecto_id), None)


def load_profile(path: Path) -> Profile:
    with path.open(encoding="utf-8") as f:
        return Profile.model_validate(yaml.safe_load(f))


# ─── Versión en texto ───────────────────────────────────────────────────────
# El texto debe ser determinista: si cambia un solo carácter entre requests,
# se pierde el prompt caching del proveedor.


def _periodo(inicio: str | None, fin: str | None, actual: str) -> str:
    return f"{inicio or '?'} – {fin or actual}"


def render_contacto(p: Profile) -> str:
    datos = p.contacto.model_dump(exclude_none=True)
    return "\n".join(f"- {k}: {v}" for k, v in datos.items()) or "- (sin datos de contacto)"


def render_experiencia(p: Profile) -> str:
    bloques = []
    for e in p.experiencia:
        cabecera = f"### {e.puesto} — {e.empresa} ({_periodo(e.inicio, e.fin, 'actual')})"
        if e.modalidad:
            cabecera += f" · {e.modalidad}"
        bloque = f"{cabecera}\n{e.descripcion}"
        if e.tecnologias:
            bloque += f"\nTecnologías: {', '.join(e.tecnologias)}"
        bloques.append(bloque)
    return "\n\n".join(bloques) or "(sin experiencia registrada)"


def render_educacion(p: Profile) -> str:
    lineas = []
    for ed in p.educacion:
        linea = f"- {ed.grado} — {ed.institucion} ({_periodo(ed.inicio, ed.fin, 'en curso')})"
        if ed.notas:
            linea += f". {ed.notas}"
        lineas.append(linea)
    return "\n".join(lineas) or "(sin educación registrada)"


def render_habilidades(p: Profile) -> str:
    return "\n".join(f"- {cat}: {', '.join(items)}" for cat, items in p.habilidades.items())


def render_proyecto(pr: Proyecto, *, con_detalle: bool) -> str:
    cabecera = f"### {pr.nombre} (id: {pr.id})"
    meta = [x for x in (pr.contexto, pr.fecha) if x]
    if meta:
        cabecera += f" — {' · '.join(meta)}"
    bloque = f"{cabecera}\n{pr.resumen}"
    if pr.tecnologias:
        bloque += f"\nTecnologías: {', '.join(pr.tecnologias)}"
    if pr.enlaces:
        bloque += f"\nEnlaces: {', '.join(pr.enlaces)}"
    if con_detalle and pr.detalle:
        bloque += f"\n\nDetalle:\n{pr.detalle.strip()}"
    return bloque


def render_lista(items: list[str], vacio: str) -> str:
    return "\n".join(f"- {i}" for i in items) or vacio


def render_profile(p: Profile, *, con_detalle: bool) -> str:
    """Convertir el perfil completo a texto. Sin detalle, los proyectos quedan solo con su resumen."""
    proyectos = "\n\n".join(render_proyecto(pr, con_detalle=con_detalle) for pr in p.proyectos)
    secciones = [
        f"# {p.nombre}\n{p.titular}" + (f"\nUbicación: {p.ubicacion}" if p.ubicacion else ""),
        f"## Resumen\n{p.resumen.strip()}",
        f"## Contacto\n{render_contacto(p)}",
        f"## Experiencia\n{render_experiencia(p)}",
        f"## Educación\n{render_educacion(p)}",
        f"## Habilidades\n{render_habilidades(p)}",
        f"## Proyectos\n{proyectos or '(sin proyectos registrados)'}",
        f"## Logros\n{render_lista(p.logros, '(sin logros registrados)')}",
        f"## Cursos relevantes\n{render_lista(p.cursos, '(sin cursos registrados)')}",
        f"## Idiomas\n{render_lista(p.idiomas, '(no hay información de idiomas)')}",
    ]
    return "\n\n".join(secciones)
