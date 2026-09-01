"""Health checks for Render liveness and readiness probes."""

from __future__ import annotations

import app.redis_client as redis_client


async def check_readiness() -> bool:
    """Return whether the shared session/cache dependency is reachable."""
    if redis_client.redis is None:
        return False
    try:
        return bool(await redis_client.redis.ping())
    except Exception:
        return False
