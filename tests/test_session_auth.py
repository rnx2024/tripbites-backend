from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.session.session_auth import sign_session, verify_session


def test_session_token_round_trip() -> None:
    session_id = "session-123"

    verify_session(session_id, sign_session(session_id))


def test_expired_session_token_is_rejected() -> None:
    session_id = "session-expired"
    token = sign_session(session_id)

    with patch("app.session.session_auth.settings.session_token_ttl_seconds", -1), pytest.raises(HTTPException) as raised:
        verify_session(session_id, token)

    assert raised.value.status_code == 401
    assert raised.value.detail == "Invalid or expired session token"
