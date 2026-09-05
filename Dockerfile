# ---- Builder: resolve dependencies (wheels only, no local project build) ----
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /usr/local/bin/uv

WORKDIR /app

# Install third-party dependencies pinned exactly as resolved in uv.lock (--frozen),
# wheels only (--no-build), skipping the dev dependency group. The local project is
# never installed via uv (--no-install-project) — the runtime stage runs it straight
# from source via PYTHONPATH instead, so --no-build applies uniformly, with no
# exception for our own code either.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-build --no-install-project

# ---- Runtime: no uv/pip/build tooling, non-root ----
FROM python:3.11-slim AS runtime

# Drop the base image's own system-level pip bootstrap (pip/setuptools/wheel/
# jaraco-context) and create the non-root runtime user in one layer. The app runs
# entirely from the self-contained venv copied below and never invokes system pip.
RUN pip uninstall -y jaraco-context wheel setuptools pip \
    && groupadd --system app && useradd --system --no-create-home --gid app app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --chown=app:app app ./app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER app

# Uvicorn binds to $PORT if Render (or another host) sets it, falling back to 8080 locally.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8080') + '/health/live', timeout=3)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
