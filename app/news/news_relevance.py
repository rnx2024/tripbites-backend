from __future__ import annotations

import re
from typing import Any

_PLACE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_RAW_URL_RE = re.compile(r"https?://[^\s)\]]+")
_HIGH_IMPACT_TERMS = (
    "suspend",
    "suspension",
    "closed",
    "closure",
    "executive order",
    "evacuation",
    "cancelled",
    "canceled",
    "class",
    "work in public offices",
)


def _tokens(value: str | None) -> set[str]:
    return set(_PLACE_TOKEN_RE.findall((value or "").casefold()))


def normalize_place(place: str) -> str:
    return " ".join(_PLACE_TOKEN_RE.findall(place.casefold()))


def news_item_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(field) or "").strip() for field in ("title", "snippet") if str(item.get(field) or "").strip()
    )


def news_item_mentions_place(item: dict[str, Any], place: str) -> bool:
    place_tokens = _tokens(place)
    if not place_tokens:
        return False

    article_tokens = _tokens(news_item_text(item))
    if place_tokens <= article_tokens:
        return True

    # City suffixes are commonly omitted in headlines ("Vigan" vs "Vigan City").
    reduced_place_tokens = place_tokens - {"city", "municipality", "town"}
    if not reduced_place_tokens:
        return False
    if reduced_place_tokens <= article_tokens:
        return True

    # Composite Philippine place names are often shortened to their municipality
    # in headlines (for example, "San Fernando" for "San Fernando, La Union").
    # Require at least two meaningful matching tokens to avoid accepting a generic
    # single-token overlap as geographic proof.
    return len(reduced_place_tokens) >= 3 and len(reduced_place_tokens & article_tokens) >= 2


def filter_relevant_news(items: list[dict[str, Any]], place: str) -> list[dict[str, Any]]:
    return [item for item in items if isinstance(item, dict) and news_item_mentions_place(item, place)]


def contains_high_impact_claim(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(term in lowered for term in _HIGH_IMPACT_TERMS)


def supports_high_impact_claim(answer: str, item: dict[str, Any], place: str) -> bool:
    if not news_item_mentions_place(item, place):
        return False
    if not contains_high_impact_claim(answer):
        return True
    return contains_high_impact_claim(news_item_text(item))


def sanitize_answer_links(answer: str, allowed_links: set[str]) -> str:
    def replace_markdown(match: re.Match[str]) -> str:
        label, link = match.groups()
        return match.group(0) if link in allowed_links else label

    sanitized = _MARKDOWN_LINK_RE.sub(replace_markdown, answer or "")

    def replace_raw(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        url = raw_url.rstrip(".,!?;:")
        return raw_url if url in allowed_links else ""

    return _RAW_URL_RE.sub(replace_raw, sanitized)
