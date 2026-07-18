from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from app.settings import settings

log = logging.getLogger(__name__)

_MAX_RESULTS = 5
_TIMEOUT_SECONDS = 10.0
_RETRIES = 3


def _infer_source_name(url: str) -> str | None:
    host = urlparse(url).netloc.strip().lower()
    if not host:
        return None
    return host.removeprefix("www.")


def _normalize_tavily_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in results[:_MAX_RESULTS]:
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("content") or "").strip()
        published = str(item.get("published_date") or item.get("date") or "").strip() or None
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
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }
    if place_hint:
        payload["query"] = f"{query} {place_hint}".strip()

    last_err = ""
    for attempt in range(1, _RETRIES + 1):
        try:
            response = httpx.post(settings.tavily_search_url, json=payload, timeout=_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            results = data.get("results") or []
            return _normalize_tavily_results(results), ""
        except httpx.TimeoutException:
            last_err = "timeout"
            log.warning("Timeout from Tavily [attempt %s/%s]", attempt, _RETRIES)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            last_err = str(status)
            log.error("HTTP %s from Tavily [attempt %s/%s]", status, attempt, _RETRIES)
        except httpx.RequestError as exc:
            last_err = str(exc)
            log.error("Request error from Tavily [attempt %s/%s]: %s", attempt, _RETRIES, exc)
        except ValueError:
            last_err = "invalid_json"
            log.error("Invalid JSON from Tavily [attempt %s/%s]", attempt, _RETRIES)

    return [], last_err
