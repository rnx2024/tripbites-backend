# app/session/session_store.py
from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from redis.exceptions import RedisError

import app.redis_client as redis_client
from app.session.errors import SESSION_UNAVAILABLE_MESSAGE, SessionStoreUnavailable
from app.session.session_keys import DEFAULT_SESSION_TTL, ONE_HOUR, news_key, sess_key, to_int, weather_key

log = logging.getLogger(__name__)

_PENDING_AGENT_CONTEXT_FIELD = "pending_agent_context"
_RECENT_TURNS_FIELD = "recent_turns"
_ACTIVE_DESTINATION_FIELD = "active_destination"
_ACTIVE_ORIGIN_FIELD = "active_origin"
_MAX_RECENT_TURNS = 6

# Test override hook. Production uses redis_client.redis.
redis: Any | None = None


def _active_redis() -> Any | None:
    return redis if redis is not None else redis_client.redis


def _require_redis() -> Any:
    client = _active_redis()
    if client is None:
        raise SessionStoreUnavailable(SESSION_UNAVAILABLE_MESSAGE)
    return client


async def _fetch_field(session_id: str, field: str, *, log_context: str) -> str | None:
    client = _require_redis()
    try:
        return await client.hget(sess_key(session_id), field)
    except RedisError as exc:
        log.warning("Redis read failed in %s [session_id=%s]: %s", log_context, session_id, exc)
        raise SessionStoreUnavailable(SESSION_UNAVAILABLE_MESSAGE) from exc


async def _fetch_session_map(session_id: str, *, log_context: str) -> dict[str, Any]:
    client = _require_redis()
    try:
        data = await client.hgetall(sess_key(session_id))
        return data or {}
    except RedisError as exc:
        log.warning("Redis hgetall failed in %s [session_id=%s]: %s", log_context, session_id, exc)
        raise SessionStoreUnavailable(SESSION_UNAVAILABLE_MESSAGE) from exc


async def _write_field(
    session_id: str,
    field: str,
    value: str | None,
    *,
    ttl_seconds: int,
    log_context: str,
) -> None:
    if value is not None:
        value = str(value)

    client = _require_redis()
    sk = sess_key(session_id)
    try:
        if value:
            await client.hset(sk, mapping={field: value})
            await client.expire(sk, ttl_seconds)
        else:
            await client.hdel(sk, field)
    except RedisError as exc:
        log.warning("Redis write failed in %s [session_id=%s]: %s", log_context, session_id, exc)
        raise SessionStoreUnavailable(SESSION_UNAVAILABLE_MESSAGE) from exc


def _decode_pending_context(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        return None
    return {str(key): str(value) for key, value in data.items() if value is not None}


def _decode_recent_turns(raw: str | None) -> list[dict[str, str]]:
    if not raw:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        return []
    turns: list[dict[str, str]] = []
    for item in data[-_MAX_RECENT_TURNS:]:
        if not isinstance(item, dict):
            continue
        turns.append(
            {
                "user": str(item.get("user") or ""),
                "assistant": str(item.get("assistant") or ""),
            }
        )
    return turns


async def _resolve_compute_value(compute_fn: Callable[[], Awaitable[str] | str]) -> str:
    value = compute_fn()
    return await value if inspect.isawaitable(value) else value


async def get_session_state(session_id: str) -> dict[str, Any]:
    return await _fetch_session_map(session_id, log_context="get_session_state")


async def should_include(session_id: str, force_weather: bool, force_news: bool) -> tuple[bool, bool]:
    data = await _fetch_session_map(session_id, log_context="should_include")
    now = int(time.time())
    last_w = to_int((data or {}).get("last_weather_sent_at"), 0)
    last_n = to_int((data or {}).get("last_news_sent_at"), 0)

    include_weather = force_weather or (now - last_w >= ONE_HOUR)
    include_news = force_news or (now - last_n >= ONE_HOUR)
    return include_weather, include_news


async def mark_sent(
    session_id: str,
    *,
    weather_sent: bool,
    news_sent: bool,
    user_message: str | None = None,
    agent_reply: str | None = None,
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
    now = int(time.time())
    mapping: dict[str, str] = {}

    if weather_sent:
        mapping["last_weather_sent_at"] = str(now)
    if news_sent:
        mapping["last_news_sent_at"] = str(now)

    mapping["last_chat_sent_at"] = str(now)

    if user_message:
        mapping["last_user_message"] = user_message[:500]
    if agent_reply:
        mapping["last_agent_reply"] = agent_reply[:1000]

    if user_message or agent_reply:
        recent_turns = await get_recent_turns(session_id)
        recent_turns.append(
            {
                "user": (user_message or "")[:500],
                "assistant": (agent_reply or "")[:1000],
            }
        )
        mapping[_RECENT_TURNS_FIELD] = json.dumps(recent_turns[-_MAX_RECENT_TURNS:], ensure_ascii=True)

    client = _require_redis()
    sk = sess_key(session_id)
    try:
        await client.hset(sk, mapping=mapping)
        await client.expire(sk, ttl_seconds)
    except RedisError as exc:
        log.warning("Redis write failed in mark_sent [session_id=%s]: %s", session_id, exc)
        raise SessionStoreUnavailable(SESSION_UNAVAILABLE_MESSAGE) from exc


async def set_pending_journey_question(
    session_id: str,
    question: str | None,
    *,
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
    value = question[:500] if question else None
    await _write_field(
        session_id,
        "pending_journey_question",
        value,
        ttl_seconds=ttl_seconds,
        log_context="set_pending_journey_question",
    )


async def set_pending_agent_context(
    session_id: str,
    context: dict[str, str] | None,
    *,
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
    try:
        payload = json.dumps(context, ensure_ascii=True) if context else None
    except (TypeError, ValueError) as exc:
        log.warning("Redis write failed in set_pending_agent_context [session_id=%s]: %s", session_id, exc)
        return

    await _write_field(
        session_id,
        _PENDING_AGENT_CONTEXT_FIELD,
        payload,
        ttl_seconds=ttl_seconds,
        log_context="set_pending_agent_context",
    )


async def set_active_destination(
    session_id: str,
    place: str | None,
    *,
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
    value = place[:200] if place else None
    await _write_field(
        session_id,
        _ACTIVE_DESTINATION_FIELD,
        value,
        ttl_seconds=ttl_seconds,
        log_context="set_active_destination",
    )


async def set_active_origin(
    session_id: str,
    origin: str | None,
    *,
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
    value = origin[:200] if origin else None
    await _write_field(
        session_id,
        _ACTIVE_ORIGIN_FIELD,
        value,
        ttl_seconds=ttl_seconds,
        log_context="set_active_origin",
    )


async def get_pending_journey_question(session_id: str) -> str | None:
    value = await _fetch_field(session_id, "pending_journey_question", log_context="get_pending_journey_question")
    return value or None


async def get_active_destination(session_id: str) -> str | None:
    value = await _fetch_field(session_id, _ACTIVE_DESTINATION_FIELD, log_context="get_active_destination")
    return str(value).strip() if value else None


async def get_active_origin(session_id: str) -> str | None:
    value = await _fetch_field(session_id, _ACTIVE_ORIGIN_FIELD, log_context="get_active_origin")
    return str(value).strip() if value else None


async def get_pending_agent_context(session_id: str) -> dict[str, str] | None:
    raw = await _fetch_field(session_id, _PENDING_AGENT_CONTEXT_FIELD, log_context="get_pending_agent_context")
    try:
        return _decode_pending_context(raw)
    except (ValueError, TypeError) as exc:
        log.warning("Redis read failed in get_pending_agent_context [session_id=%s]: %s", session_id, exc)
        return None


async def get_recent_turns(session_id: str) -> list[dict[str, str]]:
    raw = await _fetch_field(session_id, _RECENT_TURNS_FIELD, log_context="get_recent_turns")
    try:
        return _decode_recent_turns(raw)
    except (ValueError, TypeError) as exc:
        log.warning("Redis read failed in get_recent_turns [session_id=%s]: %s", session_id, exc)
        return []


async def mark_tools_called(
    session_id: str,
    *,
    tool_names: Iterable[str],
    user_message: str | None = None,
    agent_reply: str | None = None,
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
    tset = {str(t or "").strip() for t in (tool_names or []) if str(t or "").strip()}
    await mark_sent(
        session_id,
        weather_sent=("weather_tool" in tset),
        news_sent=("news_tool" in tset or "news_search_tool" in tset),
        user_message=user_message,
        agent_reply=agent_reply,
        ttl_seconds=ttl_seconds,
    )


async def get_last_exchange(session_id: str) -> tuple[str | None, str | None]:
    last_user = await _fetch_field(session_id, "last_user_message", log_context="get_last_exchange")
    last_reply = await _fetch_field(session_id, "last_agent_reply", log_context="get_last_exchange")
    return last_user, last_reply


async def get_last_sent_timestamps(session_id: str) -> tuple[int, int, int]:
    w = await _fetch_field(session_id, "last_weather_sent_at", log_context="get_last_sent_timestamps")
    n = await _fetch_field(session_id, "last_news_sent_at", log_context="get_last_sent_timestamps")
    c = await _fetch_field(session_id, "last_chat_sent_at", log_context="get_last_sent_timestamps")
    return to_int(w, 0), to_int(n, 0), to_int(c, 0)


async def ensure_session_store_ready() -> None:
    if _active_redis() is not None:
        return
    try:
        await redis_client.init_redis()
    except Exception as exc:
        log.warning("Session store init failed: %s", exc)
        raise SessionStoreUnavailable(SESSION_UNAVAILABLE_MESSAGE) from exc
    _require_redis()


async def get_or_set(
    key: str,
    ttl_seconds: int,
    compute_fn: Callable[[], Awaitable[str] | str],
) -> str:
    redis_conn = _active_redis()
    if redis_conn is None:
        return await _resolve_compute_value(compute_fn)

    try:
        cached = await redis_conn.get(key)
        if cached is not None:
            return cached
    except RedisError as exc:
        log.warning("Redis get failed in get_or_set [key=%s]: %s", key, exc)
        return await _resolve_compute_value(compute_fn)

    value = await _resolve_compute_value(compute_fn)
    try:
        await redis_conn.set(key, value, ex=ttl_seconds)
    except RedisError as exc:
        log.warning("Redis set failed in get_or_set [key=%s]: %s", key, exc)
    return value


async def prepare_weather_news(
    *,
    session_id: str,
    user_text: str,
    location: str,
    fetch_weather_fn: Callable[[], Awaitable[str] | str],
    fetch_news_fn: Callable[[], Awaitable[str] | str],
) -> tuple[str, str, bool, bool]:
    text_lc = (user_text or "").lower()
    force_weather = "weather" in text_lc
    force_news = "news" in text_lc

    include_weather, include_news = await should_include(session_id, force_weather, force_news)

    weather_text = ""
    news_text = ""

    if include_weather:
        weather_text = await get_or_set(weather_key(location), ONE_HOUR, fetch_weather_fn)

    if include_news:
        news_text = await get_or_set(news_key(location), ONE_HOUR, fetch_news_fn)

    return weather_text, news_text, include_weather, include_news
