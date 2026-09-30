# Distillery: API + built frontend in one image (same origin, so VITE_API_BASE stays empty).
# The frontend alone can be hosted elsewhere (Vercel): see docs/DEPLOY.md.

# ---- stage 1: build the frontend ----
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY frontend/ ./
# Only frontend/ is in this stage: `npm run build` uses tsconfig.build.json, which leaves out the
# tests (they import ../../docs/fixtures, which is not copied here).
# Public, build-time only. Empty = same origin. Never put a secret in a VITE_* variable.
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# ---- stage 2: python runtime ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISTILLERY_HOME=/data \
    DISTILLERY_REPLAY_DIR=/app/replay \
    PORT=8000
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY docs/fixtures ./docs/fixtures
# Editable install so the server finds docs/fixtures and frontend/dist relative to /app.
# fastapi and uvicorn come from pyproject.toml dependencies.
RUN pip install -e .
COPY --from=web /web/dist ./frontend/dist
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data /app/replay \
    && chown -R app:app /data /app/replay
USER app
EXPOSE 8000
# No secrets are baked in: pass NEBIUS_API_KEY, DISTILLERY_ADMIN_TOKEN etc. at run time.
# Without them the server starts in replay-only mode.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000'), timeout=4)"]
CMD ["sh", "-c", "exec python -m distillery serve --host 0.0.0.0 --port ${PORT}"]
