# app/tooling/sync_cache.py
from __future__ import annotations

import json
from typing import Any, cast

import redis
from redis.exceptions import RedisError

from app.settings import settings

CACHE_TTL_SECONDS_DEFAULT = 3600

_sync_redis: redis.Redis | None = None


def _get_sync_redis() -> redis.Redis | None:
    """
    Sync Redis client for LangGraph sync tool calls.
    Uses REDIS_URL. Safe to return None if misconfigured.
    """
    global _sync_redis
    if _sync_redis is not None:
        return _sync_redis

    url = settings.redis_url
    if not url:
        return None

    try:
        _sync_redis = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )
        _sync_redis.ping()
        return _sync_redis
    except RedisError:
        _sync_redis = None
        return None


def cache_get_str(key: str) -> str | None:
    r = _get_sync_redis()
    if r is None:
        return None
    try:
        # redis-py's stub returns Awaitable[Any] | Any because Redis supports both sync and
        # async usage; decode_responses=True on this client guarantees a str at runtime here.
        v = cast("str | None", r.get(key))
        return v if v is not None else None
    except RedisError:
        return None


def cache_set_str(key: str, value: str, ttl: int = CACHE_TTL_SECONDS_DEFAULT) -> None:
    r = _get_sync_redis()
    if r is None:
        return
    try:
        r.set(key, value, ex=ttl)
    except RedisError:
        return


def cache_get_json(key: str) -> Any | None:
    raw = cache_get_str(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def cache_set_json(key: str, obj: Any, ttl: int = CACHE_TTL_SECONDS_DEFAULT) -> None:
    try:
        raw = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return
    cache_set_str(key, raw, ttl=ttl)
