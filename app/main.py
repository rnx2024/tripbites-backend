# main.py
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.redis_client import close_redis, init_redis
from app.routes import router as api_router
from app.settings import settings
from app.tooling.ratelimit import limiter

is_production = os.getenv("ENV", "").lower() == "production"


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
