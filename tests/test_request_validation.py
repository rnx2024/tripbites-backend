from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.session.session_auth import sign_session
from app.settings import settings
from app.validation import MAX_PLACE_LENGTH, MAX_QUESTION_LENGTH, MAX_REQUEST_BODY_BYTES


def _client() -> TestClient:
    return TestClient(app)


def test_chat_rejects_overlong_question_before_calling_agent() -> None:
    session_id = "validation-test-session"
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        patch("app.routes.run_agent") as run_agent,
        _client() as client,
    ):
        response = client.post(
            "/chat",
            headers={
                "x-api-key": settings.api_key,
                "x-session-id": session_id,
                "x-session-token": sign_session(session_id),
            },
            json={"place": "Vigan", "question": "x" * (MAX_QUESTION_LENGTH + 1)},
        )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["details"]["question"] == (
        f"String should have at most {MAX_QUESTION_LENGTH} characters"
    )
    run_agent.assert_not_called()


def test_weather_rejects_blank_and_overlong_place_before_calling_provider() -> None:
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        patch("app.routes.get_weather_line") as get_weather_line,
        _client() as client,
    ):
        blank_response = client.get(
            "/weather",
            params={"place": "   "},
            headers={"x-api-key": settings.api_key},
        )
        long_response = client.get(
            "/weather",
            params={"place": "x" * (MAX_PLACE_LENGTH + 1)},
            headers={"x-api-key": settings.api_key},
        )

    assert blank_response.status_code == 422
    assert long_response.status_code == 422
    get_weather_line.assert_not_called()


def test_weather_strips_valid_place_before_calling_provider() -> None:
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        patch("app.routes.get_weather_line", return_value=("Sunny", "")) as get_weather_line,
        _client() as client,
    ):
        response = client.get(
            "/weather",
            params={"place": "  Vigan  "},
            headers={"x-api-key": settings.api_key},
        )

    assert response.status_code == 200
    get_weather_line.assert_called_once_with("Vigan")
    assert response.json()["place"] == "Vigan"


def test_request_body_limit_returns_413_before_authentication() -> None:
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        _client() as client,
    ):
        response = client.post(
            "/chat",
            content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "REQUEST_TOO_LARGE",
            "message": "Request is too large. Please shorten your message.",
        }
    }


def test_invalid_json_returns_400() -> None:
    with (
        patch("app.main.init_redis", new=AsyncMock(return_value=None)),
        patch("app.main.close_redis", new=AsyncMock(return_value=None)),
        _client() as client,
    ):
        response = client.post(
            "/chat",
            content=b"not-json",
            headers={
                "content-type": "application/json",
                "x-api-key": settings.api_key,
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "BAD_REQUEST",
            "message": "Invalid JSON body.",
        }
    }
