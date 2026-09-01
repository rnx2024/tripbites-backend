from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import allowed_origins, app


def test_configured_cors_origins_are_normalized() -> None:
    assert allowed_origins
    assert all(not origin.endswith("/") for origin in allowed_origins)


def test_liveness_does_not_require_redis() -> None:
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        TestClient(app) as client,
    ):
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_readiness_reports_redis_unavailable() -> None:
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        patch("app.redis_client.redis", None),
        TestClient(app) as client,
    ):
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
