# app/redis_client.py
"""
Redis client lifecycle for SmartNews.

- One Redis connection per FastAPI process
- Initialized on startup
- Closed on shutdown
- Used ONLY for short-lived session + cache data
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.settings import settings

log = logging.getLogger(__name__)

redis: Redis | None = None


def _safe_redis_target(redis_url: str) -> str:
    try:
        parsed = urlparse(redis_url)
        host = parsed.hostname or "unknown-host"
        scheme = parsed.scheme or "redis"
        port = parsed.port or (6380 if scheme == "rediss" else 6379)
        return f"{scheme}://{host}:{port}"
    except Exception:
        return "redis://unknown-host"


async def init_redis() -> None:
    """
    Initialize Redis client once per process.
    Must be called on FastAPI startup.
    """
    global redis

    if redis is not None:
        return

    redis_url = settings.redis_url
    if not redis_url:
        log.warning("Redis required but REDIS_URL is missing")
        raise RuntimeError("REDIS_URL is not set")
    target = _safe_redis_target(redis_url)

    client = Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
    )

    try:
        # Health check connection eagerly to avoid hidden runtime failures.
        await client.ping()
    except (RedisError, OSError) as exc:
        await client.aclose()
        log.warning("Redis connection failed [target=%s, error=%s]: %s", target, type(exc).__name__, exc)
        raise RuntimeError("Redis connection failed") from exc

    redis = client
    log.info("Redis connected [target=%s]", target)


async def close_redis() -> None:
    """
    Cleanly close Redis connection.
    Must be called on FastAPI shutdown.
    """
    global redis

    if redis is not None:
        await redis.aclose()
        redis = None
