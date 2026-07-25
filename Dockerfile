FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies pinned exactly as resolved in uv.lock (--frozen), wheels only
# (--no-build), skipping the dev dependency group.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-build

# Copy source
COPY app ./app

# Uvicorn binds to $PORT if Render (or another host) sets it, falling back to 8080 locally.
CMD ["sh", "-c", "uv run --frozen --no-dev uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
