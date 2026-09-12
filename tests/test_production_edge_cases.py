from __future__ import annotations

import logging
from argparse import Namespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from scripts.load_test import RequestResult, _percentile, _summarize


def test_request_completion_log_contains_safe_timing_fields(caplog) -> None:
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        caplog.at_level(logging.INFO),
        TestClient(app) as client,
    ):
        response = client.get("/health/live", headers={"x-request-id": "edge-test-1"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "edge-test-1"
    assert "request.completed" in caplog.text
    assert "duration_ms" in caplog.text
    assert "edge-test-1" in caplog.text


def test_load_test_percentile_and_summary() -> None:
    results = [
        RequestResult(10.0, 200, None),
        RequestResult(20.0, 200, None),
        RequestResult(30.0, 503, "http_5xx"),
        RequestResult(40.0, None, "timeout"),
    ]
    args = Namespace(url="http://127.0.0.1:8000", endpoint="health/live")

    assert _percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 25.0
    report = _summarize(results, 1.0, args)

    assert report["total_requests"] == 4
    assert report["successful_requests"] == 2
    assert report["failed_requests"] == 2
    assert report["error_categories"] == {"http_5xx": 1, "timeout": 1}


def test_liveness_remains_available_without_redis() -> None:
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        TestClient(app) as client,
    ):
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_returns_unavailable_without_redis() -> None:
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        patch("app.redis_client.redis", None),
        TestClient(app) as client,
    ):
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
