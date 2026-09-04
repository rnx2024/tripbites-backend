from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from app.provider_schemas import as_list, as_mapping, bounded_text
from app.settings import settings
from app.tooling.retry_policy import build_http_retry

log = logging.getLogger(__name__)

_MAX_RESULTS = 5
_TIMEOUT_SECONDS = 10.0
_RETRIES = 3
_TIME_RANGE = "week"


def _infer_source_name(url: str) -> str | None:
    host = urlparse(url).netloc.strip().lower()
    if not host:
        return None
    return host.removeprefix("www.")


def _normalize_tavily_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in results[:_MAX_RESULTS]:
        if not isinstance(item, dict):
            continue
        url = bounded_text(item.get("url"), maximum=2048) or ""
        title = bounded_text(item.get("title"), maximum=500) or ""
        snippet = bounded_text(item.get("content"), maximum=2000) or ""
        published = bounded_text(item.get("published_date") or item.get("date"), maximum=100)
        normalized.append(
            {
                "title": title or "Untitled",
                "source": _infer_source_name(url),
                "date": published,
                "link": url or None,
                "snippet": snippet or None,
            }
        )
    return normalized


def search_tavily(query: str, place_hint: str | None = None) -> tuple[list[dict[str, Any]], str]:
    if not settings.tavily_api:
        return [], "missing_tavily_api"

    payload = {
        "api_key": settings.tavily_api,
        "query": query,
        "topic": "general",
        "search_depth": "advanced",
        "max_results": _MAX_RESULTS,
        "time_range": _TIME_RANGE,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }
    if place_hint:
        payload["query"] = f"{query} {place_hint}".strip()

    @build_http_retry(max_attempts=_RETRIES)
    def _post() -> httpx.Response:
        response = httpx.post(settings.tavily_search_url, json=payload, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response

    try:
        response = _post()
    except httpx.TimeoutException:
        log.warning("Timeout from Tavily after %s attempt(s)", _RETRIES)
        return [], "timeout"
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        log.error("HTTP %s from Tavily after %s attempt(s)", status, _RETRIES)
        return [], str(status)
    except httpx.RequestError:
        log.exception("Request error from Tavily after %s attempt(s)", _RETRIES)
        return [], "request_failed"

    try:
        data = response.json()
    except ValueError:
        log.error("Invalid JSON from Tavily")
        return [], "invalid_json"

    data = as_mapping(data)
    if data is None:
        return [], "invalid_response"
    return _normalize_tavily_results(as_list(data.get("results"))), ""
