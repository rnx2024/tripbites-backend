from __future__ import annotations

from typing import Any

from app.news.serpapi_news_fetcher import fetch_news_items
from app.news.tavily_search_fetcher import search_tavily


def get_news_items(place: str) -> tuple[list[dict[str, Any]], str]:
    """
    Backwards-compatible facade for news retrieval.
    """
    return fetch_news_items(place)


def search_news(query: str, place_hint: str | None = None) -> tuple[list[dict[str, Any]], str]:
    """
    Targeted follow-up search for missing facts in QA turns.
    This is intentionally separate from the destination news feed.
    """
    return search_tavily(query, place_hint)
