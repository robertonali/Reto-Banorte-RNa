# Imagen de producción del agente de CV (la que construye Render).
FROM python:3.13-slim

# Usar uv para instalar exactamente las versiones fijadas en uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Instalar primero las dependencias: esta capa se reutiliza mientras uv.lock no cambie.
# --no-dev deja fuera pytest; MLflow (grupo evals) tampoco entra a la imagen.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

# Correr el servicio sin privilegios de root.
RUN useradd --create-home --uid 1000 agente
USER agente

# NO meter el perfil en la imagen: en Render se sube como Secret File.
ENV PATH="/app/.venv/bin:$PATH" \
    ENVIRONMENT=production \
    PROFILE_PATH=/etc/secrets/profile.yaml

EXPOSE 10000
# Render define PORT; --proxy-headers toma la IP real del cliente (la usa el rate limit).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips='*' --no-access-log"]
