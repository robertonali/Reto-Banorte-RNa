"""Bloquear commits que incluyan datos personales.

Revisar los archivos en staging (o todo el repo con --all) y fallar si aparece:
- un archivo que nunca debe subirse (private/, .env, *.tex, notebooks);
- un dato de contacto de private/profile.yaml (correo, teléfono, LinkedIn, YouTube);
- cualquier valor listado en private/sensitive.txt (uno por línea).

Usar solo la librería estándar, para que el hook funcione sin instalar nada.
Instalar el hook con: git config core.hooksPath .githooks
"""

import re
import subprocess
import sys
from pathlib import Path

BLOCKED = re.compile(r"(^|/)(private/|\.env$|\.env\.(?!example$))|\.tex$|\.ipynb$")
CONTACT_KEYS = ("email", "telefono", "linkedin", "youtube")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def sensitive_values() -> set[str]:
    values: set[str] = set()
    profile = Path("private/profile.yaml")
    if profile.exists():
        text = profile.read_text(encoding="utf-8")
        for key in CONTACT_KEYS:
            match = re.search(rf"^\s+{key}:\s*[\"']?([^\"'\n#]+)", text, re.MULTILINE)
            if match and match.group(1).strip():
                values.add(match.group(1).strip())
    extra = Path("private/sensitive.txt")
    if extra.exists():
        values |= {line.strip() for line in extra.read_text(encoding="utf-8").splitlines() if line.strip()}
    return values


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def main() -> int:
    scan_all = "--all" in sys.argv
    if scan_all:
        files = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    else:
        files = git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines()

    values = sensitive_values()
    # Buscar los teléfonos solo por dígitos, para atrapar cualquier formato (+52 55..., 5541...).
    phones = {digits(v)[-10:] for v in values if len(digits(v)) >= 10}
    problems = []
    for name in files:
        if BLOCKED.search(name):
            problems.append(f"{name}: este archivo nunca debe subirse")
            continue
        try:
            content = Path(name).read_text(encoding="utf-8") if scan_all else git("show", f":{name}")
        except (UnicodeDecodeError, subprocess.CalledProcessError, FileNotFoundError):
            continue  # saltar binarios o archivos borrados
        lowered = content.lower()
        for value in values:
            if value.lower() in lowered:
                problems.append(f"{name}: contiene un dato sensible ({value[:4]}…)")
        content_digits = digits(content)
        for phone in phones:
            if phone in content_digits:
                problems.append(f"{name}: contiene un teléfono")

    if problems:
        print("Commit bloqueado: se encontraron datos personales.\n", file=sys.stderr)
        for p in dict.fromkeys(problems):  # sin repetidos, en orden
            print(f"  - {p}", file=sys.stderr)
        print("\nQuítalos (o muévelos a private/) y vuelve a intentar.", file=sys.stderr)
        return 1
    print(f"check_sensitive: {len(files)} archivos revisados, sin datos personales.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
