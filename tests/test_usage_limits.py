from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.tooling.usage_limits import enforce_chat_quota


def test_chat_quota_sets_expiry_on_first_request() -> None:
    redis = AsyncMock()
    redis.incr.return_value = 1

    with patch("app.redis_client.redis", redis):
        asyncio.run(enforce_chat_quota("session-1"))

    redis.incr.assert_awaited_once_with("quota:chat:session-1")
    redis.expire.assert_awaited_once()


def test_chat_quota_rejects_after_limit() -> None:
    redis = AsyncMock()
    redis.incr.return_value = 31

    def invoke_quota() -> None:
        asyncio.run(enforce_chat_quota("session-2"))

    with patch("app.redis_client.redis", redis), pytest.raises(HTTPException) as raised:
        invoke_quota()

    assert raised.value.status_code == 429
    assert raised.value.headers == {"Retry-After": "3600"}


def test_chat_quota_fails_open_when_redis_is_unavailable() -> None:
    with patch("app.redis_client.redis", None):
        asyncio.run(enforce_chat_quota("session-3"))
