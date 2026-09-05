# app/session_auth.py
from __future__ import annotations

from fastapi import Header, HTTPException
from itsdangerous import BadSignature, URLSafeTimedSerializer

from app.settings import settings

_CURRENT_SESSION_SALT = "tripbites-session-v1"
_serializer = URLSafeTimedSerializer(secret_key=settings.session_secret, salt=_CURRENT_SESSION_SALT)


def sign_session(session_id: str) -> str:
    return _serializer.dumps({"sid": session_id})


def verify_session(session_id: str, session_token: str) -> None:
    payload = None
    try:
        payload = _serializer.loads(session_token, max_age=settings.session_token_ttl_seconds)
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid or expired session token") from None

    if not isinstance(payload, dict) or payload.get("sid") != session_id:
        raise HTTPException(status_code=401, detail="Session token mismatch")


def require_session(
    x_session_id: str = Header(..., alias="x-session-id"),
    x_session_token: str = Header(..., alias="x-session-token"),
) -> str:
    verify_session(x_session_id, x_session_token)
    return x_session_id
