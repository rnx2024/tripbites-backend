# main.py
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.settings import settings
from app.routes import router as api_router
from app.tooling.ratelimit import limiter
from app.redis_client import init_redis, close_redis


is_production = os.getenv("ENV", "").lower() == "production"

app = FastAPI(
    title="TripBites API",
    description="Travel intelligence backend for destination briefs, local conditions, and disruption-aware city updates.",
    version="0.2.0",
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
    openapi_url=None if is_production else "/openapi.json",
)

# ---------------------------
# SlowAPI (rate limiting)
# ---------------------------
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origins = [origin.strip() for origin in settings.frontend_cors_origin.split(',')]

# ---------------------------------------------------------
# CORS CONFIGURATION
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Mount API routes AFTER adding CORS
# ---------------------------------------------------------
app.include_router(api_router)


@app.on_event("startup")
async def _startup():
    await init_redis()


@app.on_event("shutdown")
async def _shutdown():
    await close_redis()


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
