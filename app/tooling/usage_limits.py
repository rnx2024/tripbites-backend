"""Best-effort Redis-backed quotas for expensive user-facing operations."""

from __future__ import annotations

from fastapi import HTTPException

import app.redis_client as redis_client

CHAT_WINDOW_SECONDS = 3600
CHAT_REQUESTS_PER_WINDOW = 30


async def enforce_chat_quota(session_id: str) -> None:
    """Reject excessive chat requests for one signed session."""
    client = redis_client.redis
    if client is None:
        return

    key = f"quota:chat:{session_id}"
    try:
        count = int(await client.incr(key))
        if count == 1:
            await client.expire(key, CHAT_WINDOW_SECONDS)
    except Exception:
        # Availability of the chat API should not depend on the optional quota
        # counter when Redis is degraded; SlowAPI still provides the edge limit.
        return

    if count > CHAT_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=429,
            detail="Chat usage limit reached. Please try again later.",
            headers={"Retry-After": str(CHAT_WINDOW_SECONDS)},
        )
