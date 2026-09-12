"""Health checks for Render liveness and readiness probes."""

from __future__ import annotations

import inspect

import app.redis_client as redis_client


async def check_readiness() -> bool:
    """Return whether the shared session/cache dependency is reachable."""
    if redis_client.redis is None:
        return False
    try:
        result = redis_client.redis.ping()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception:
        return False
