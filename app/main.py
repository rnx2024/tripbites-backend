# main.py
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response

from app.logging_config import configure_logging

configure_logging()

from app.redis_client import close_redis, init_redis  # noqa: E402
from app.routes import router as api_router  # noqa: E402
from app.settings import settings  # noqa: E402
from app.tooling.ratelimit import limiter  # noqa: E402

is_production = os.getenv("ENV", "").lower() == "production"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis()
    yield
    await close_redis()


app = FastAPI(
    title="TripBites API",
    description=(
        "Travel intelligence backend for destination briefs, local conditions, and disruption-aware city updates."
    ),
    version="0.2.0",
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
    openapi_url=None if is_production else "/openapi.json",
    lifespan=lifespan,
)

# ---------------------------
# SlowAPI (rate limiting)
# ---------------------------
app.state.limiter = limiter
# slowapi's handler is typed for RateLimitExceeded only, narrower than Starlette's generic
# Exception handler signature; Starlette dispatches by the registered exception type, so this
# is safe at runtime.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

allowed_origins = [origin.strip() for origin in settings.frontend_cors_origin.split(",")]

# ---------------------------------------------------------
# CORS CONFIGURATION
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "x-api-key", "x-session-id", "x-session-token"],
)

# ---------------------------------------------------------
# Request-ID correlation (binds a per-request ID into every
# structlog/log line and echoes it back as X-Request-ID)
# ---------------------------------------------------------
app.add_middleware(RequestIDMiddleware)

# ---------------------------------------------------------
# Mount API routes AFTER adding CORS
# ---------------------------------------------------------
app.include_router(api_router)


@app.get("/", tags=["meta"])
async def root():
    payload = {
        "name": "TripBites API",
        "status": "ok",
        "service": "travel-intelligence",
    }
    if not is_production:
        payload["docs"] = "/docs"
    return payload
