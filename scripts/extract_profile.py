"""Generar un BORRADOR de perfil (YAML) a partir de un CV en .tex, .md, .txt o .pdf.

Uso:
    uv run python scripts/extract_profile.py private/TCS.tex
    uv run python scripts/extract_profile.py private/cv.pdf --salida private/profile.draft.yaml

Revisar el borrador a mano antes de copiarlo a private/profile.yaml. El perfil es la fuente de
verdad del agente: un error de extracción que nadie revisa se convierte en una "alucinación"
que el agente repetiría con total seguridad.
"""

import argparse
import sys
from pathlib import Path

import anthropic
import yaml
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar app/ desde scripts/

from app.config import get_settings  # noqa: E402
from app.llm.base import Usage, cost_usd  # noqa: E402
from app.profile import Contacto, Educacion, Experiencia, Profile, Proyecto  # noqa: E402



# Esquema de extracción: solo textos y listas, sin campos opcionales. Con opcionales y dicts libres,
# la gramática de structured outputs crece demasiado y la API la rechaza. Texto vacío = dato ausente.


class _Contacto(BaseModel):
    email: str
    telefono: str
    linkedin: str
    github: str
    youtube: str


class _Experiencia(BaseModel):
    id: str
    puesto: str
    empresa: str
    inicio: str
    fin: str
    modalidad: str
    descripcion: str
    tecnologias: list[str]


class _Educacion(BaseModel):
    grado: str
    institucion: str
    inicio: str
    fin: str
    notas: str


class _Proyecto(BaseModel):
    id: str
    nombre: str
    contexto: str
    fecha: str
    resumen: str
    tecnologias: list[str]
    enlaces: list[str]


class _Categoria(BaseModel):
    nombre: str
    elementos: list[str]


class Extraccion(BaseModel):
    nombre: str
    titular: str
    ubicacion: str
    resumen: str
    contacto: _Contacto
    experiencia: list[_Experiencia]
    educacion: list[_Educacion]
    habilidades: list[_Categoria]
    proyectos: list[_Proyecto]
    logros: list[str]
    cursos: list[str]
    idiomas: list[str]


def _sin_vacios(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in ("", None)}


def to_profile(x: Extraccion) -> Profile:
    return Profile(
        nombre=x.nombre,
        titular=x.titular,
        ubicacion=x.ubicacion or None,
        resumen=x.resumen,
        contacto=Contacto(**_sin_vacios(x.contacto.model_dump())),
        experiencia=[Experiencia(**_sin_vacios(e.model_dump())) for e in x.experiencia],
        educacion=[Educacion(**_sin_vacios(e.model_dump())) for e in x.educacion],
        habilidades={c.nombre: c.elementos for c in x.habilidades},
        proyectos=[Proyecto(**_sin_vacios(p.model_dump())) for p in x.proyectos],
        logros=x.logros,
        cursos=x.cursos,
        idiomas=x.idiomas,
    )


_PROMPT = """\
Extrae el perfil profesional de este CV al esquema indicado. Reglas:
- Copia los datos tal como aparecen: no traduzcas las descripciones, no resumas logros y no inventes nada.
- Fechas en español como "junio 2024". Si el puesto es actual, fin = "".
- Si un dato no aparece en el CV, usa "" o [] (por ejemplo, idiomas = [] si no se mencionan).
- ids en kebab-case (por ejemplo "empresa-puesto").
- "titular" y "resumen" solo con hechos del CV, sin adjetivos que el CV no use.

<cv>
{texto}
</cv>"""


def read_cv(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # solo hace falta para PDFs (grupo dev)

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cv", type=Path)
    parser.add_argument("--salida", type=Path, default=Path("private/profile.draft.yaml"))
    parser.add_argument("--modelo", default="claude-opus-5-5")
    args = parser.parse_args()

    if args.salida.resolve() == Path("private/profile.yaml").resolve():
        sys.exit("No sobrescribo el perfil revisado: usa otra --salida y compara a mano.")

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    response = client.messages.parse(
        model=args.modelo,
        max_tokens=16000,
        messages=[{"role": "user", "content": _PROMPT.format(texto=read_cv(args.cv))}],
        output_format=Extraccion,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        sys.exit(f"El modelo no devolvió un perfil (stop_reason={response.stop_reason}).")

    profile = to_profile(response.parsed_output)
    header = f"# BORRADOR generado por scripts/extract_profile.py desde {args.cv.name}.\n# TODO: revisarlo antes de usarlo.\n"
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(
        header + yaml.safe_dump(profile.model_dump(), allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    u = response.usage
    cost = cost_usd(args.modelo, Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens))
    print(f"Borrador escrito en {args.salida} ({len(profile.experiencia)} experiencias, "
          f"{len(profile.proyectos)} proyectos). Costo: ${cost or 0:.4f}")


if __name__ == "__main__":
    main()
