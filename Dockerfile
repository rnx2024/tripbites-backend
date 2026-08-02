# ---- Builder: resolve dependencies and build the app's venv ----
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /usr/local/bin/uv

WORKDIR /app

# Install third-party dependencies pinned exactly as resolved in uv.lock (--frozen),
# wheels only (--no-build), skipping the dev dependency group. The local project
# isn't installed yet (--no-install-project) since its source isn't copied in yet.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-build --no-install-project

# Copy source
COPY app ./app

# Install the local project now that its source is present. Third-party deps
# above stay wheel-only; this step only builds/installs our own source (no
# network fetch, no third-party build scripts).
RUN uv sync --frozen --no-dev

# ---- Runtime: no uv/pip/build tooling, non-root ----
FROM python:3.11-slim AS runtime

# Drop the base image's own system-level pip bootstrap (pip/setuptools/wheel/
# jaraco-context). The app runs entirely from the self-contained venv copied
# below and never invokes system pip at runtime.
RUN pip uninstall -y jaraco-context wheel setuptools pip

RUN groupadd --system app && useradd --system --no-create-home --gid app app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/app ./app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER app

# Uvicorn binds to $PORT if Render (or another host) sets it, falling back to 8080 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
