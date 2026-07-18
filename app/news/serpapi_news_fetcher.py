# app/news/serpapi_news_fetcher.py
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from app.http.http_utils import get_json_with_retry
from app.location.resolve_country import resolve_country_code
from app.news.serpapi_date_parser import parse_serpapi_date
from app.settings import settings

log = logging.getLogger(__name__)


def _fetch_google_news(query: str, *, gl_hint: str) -> tuple[list[dict[str, Any]], str]:
    gl_code = resolve_country_code(gl_hint)
    gl_code = gl_code.lower() if gl_code else "us"

    params = {
        "engine": "google_news",
        "q": query,
        "hl": "en",
        "gl": gl_code,
        "api_key": settings.serp_api_key,
    }

    data, err = get_json_with_retry(settings.serpapi_search_url, params)
    if err:
        log.error("SerpAPI request failed")
        return [], err

    results = data.get("news_results") or data.get("organic_results") or []

    max_age_days = 7
    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)

    filtered: list[dict[str, Any]] = []

    for item in results:
        date_raw = item.get("date") or item.get("published")
        parsed_date = parse_serpapi_date(date_raw)

        if not parsed_date or parsed_date < cutoff:
            continue

        filtered.append(
            {
                "title": item.get("title", "Untitled"),
                "source": item.get("source", {}).get("name")
                if isinstance(item.get("source"), dict)
                else item.get("source"),
                "date": parsed_date.isoformat(),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )

    filtered.sort(key=lambda x: x["date"], reverse=True)
    return filtered[:3], ""


def fetch_news_items(place: str) -> tuple[list[dict[str, Any]], str]:
    """
    Fetch recent Google News results for a selected place.
    """
    return _fetch_google_news(place, gl_hint=place)


def search_news_items(query: str, place_hint: str | None = None) -> tuple[list[dict[str, Any]], str]:
    """
    Run a targeted Google News search for follow-up questions, keeping the same
    date filtering and response shape as the default place-based search.
    """
    hint = place_hint or query
    return _fetch_google_news(query, gl_hint=hint)
