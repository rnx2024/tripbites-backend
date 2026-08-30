from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from app.validation import MAX_REQUEST_BODY_BYTES

ASGIMessage = dict[str, object]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]


class RequestBodyLimitMiddleware:
    """Reject HTTP bodies larger than the application request limit."""

    def __init__(self, app, max_bytes: int = MAX_REQUEST_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, object], receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return

            body.extend(message.get("body", b""))  # type: ignore[arg-type]
            if len(body) > self.max_bytes:
                payload = json.dumps(
                    {
                        "error": {
                            "code": "REQUEST_TOO_LARGE",
                            "message": "Request is too large. Please shorten your message.",
                        }
                    }
                ).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode("ascii")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": payload})
                return

            if not message.get("more_body", False):
                break

        sent = False

        async def replay_receive() -> ASGIMessage:
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_receive, send)

