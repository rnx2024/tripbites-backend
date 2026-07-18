# app/http_client.py
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import httpx

from app.tooling.retry_policy import build_http_retry

log = logging.getLogger(__name__)


def get_json_with_retry(
    url: str,
    params: Dict[str, Any],
    retries: int = 3,
    timeout: float = 10.0,
) -> Tuple[Dict[str, Any], str]:
    @build_http_retry(max_attempts=retries)
    def _get() -> httpx.Response:
        response = httpx.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response

    try:
        response = _get()
    except httpx.TimeoutException:
        log.warning("Timeout from %s after %s attempt(s)", url, retries)
        return {}, "timeout"
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        log.error("HTTP %s from %s after %s attempt(s)", status, url, retries)
        return {}, str(status)
    except httpx.RequestError as exc:
        log.error("Request error from %s after %s attempt(s): %s", url, retries, exc)
        return {}, str(exc)

    try:
        return response.json(), ""
    except ValueError:
        log.error("Invalid JSON from %s", url)
        return {}, "invalid_json"
