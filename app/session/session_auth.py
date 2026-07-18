# app/session_auth.py
from __future__ import annotations

from fastapi import Header, HTTPException
from itsdangerous import BadSignature, URLSafeSerializer

from app.settings import settings

_CURRENT_SESSION_SALT = "tripbites-session-v1"
_LEGACY_SESSION_SALTS = ("smartnews-session-v1",)
_serializer = URLSafeSerializer(secret_key=settings.session_secret, salt=_CURRENT_SESSION_SALT)


def _legacy_serializers() -> list[URLSafeSerializer]:
    return [URLSafeSerializer(secret_key=settings.session_secret, salt=salt) for salt in _LEGACY_SESSION_SALTS]


def sign_session(session_id: str) -> str:
    return _serializer.dumps({"sid": session_id})


def verify_session(session_id: str, session_token: str) -> None:
    payload = None
    try:
        payload = _serializer.loads(session_token)
    except BadSignature:
        for legacy_serializer in _legacy_serializers():
            try:
                payload = legacy_serializer.loads(session_token)
                break
            except BadSignature:
                continue
        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid session token") from None

    if not isinstance(payload, dict) or payload.get("sid") != session_id:
        raise HTTPException(status_code=401, detail="Session token mismatch")


def require_session(
    x_session_id: str = Header(..., alias="x-session-id"),
    x_session_token: str = Header(..., alias="x-session-token"),
) -> str:
    verify_session(x_session_id, x_session_token)
    return x_session_id
