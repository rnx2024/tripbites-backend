# app/db.py
from __future__ import annotations

import asyncio
import os

_client: object | None = None


def _load_libsql():
    try:
        import libsql  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("libsql package is not installed. Add it before using app.db.") from exc
    return libsql


def get_client():
    """
    Singleton libSQL client (remote-only).
    """
    global _client

    if _client is None:
        libsql = _load_libsql()
        libsql_url = os.environ.get("LIBSQL_URL")
        libsql_token = os.environ.get("LIBSQL_AUTH_TOKEN")

        if not libsql_url or not libsql_token:
            raise RuntimeError("LIBSQL_URL or LIBSQL_AUTH_TOKEN is not set")

        # libsql Python SDK uses a local replica file and syncs with Turso.
        # Keep your env var names unchanged.
        _client = libsql.connect(
            "smartnews.db",
            sync_url=libsql_url,
            auth_token=libsql_token,
        )

    return _client


async def execute(query: str, args: list | None = None):
    client = get_client()

    def _run():
        return client.execute(query, tuple(args or []))

    return await asyncio.to_thread(_run)


async def fetch_all(query: str, args: list | None = None):
    client = get_client()

    def _run():
        cur = client.execute(query, tuple(args or []))
        return cur.fetchall()

    return await asyncio.to_thread(_run)
