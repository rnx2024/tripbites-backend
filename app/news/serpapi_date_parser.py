# app/news/serpapi_date_parser.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

_ABSOLUTE_DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%b %d %Y",
    "%B %d %Y",
)

_ABSOLUTE_DATE_CORE_FORMATS = (
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
)


def _try_parse_absolute_date(s_clean: str) -> datetime | None:
    for fmt in _ABSOLUTE_DATE_FORMATS:
        try:
            dt = datetime.strptime(s_clean, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _try_parse_absolute_date_with_time(s_clean: str) -> datetime | None:
    try:
        core = s_clean.split(" ")[0]
    except (AttributeError, IndexError):
        return None

    for fmt in _ABSOLUTE_DATE_CORE_FORMATS:
        try:
            dt = datetime.strptime(core, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _try_parse_relative(s_lower: str, now: datetime) -> datetime | None:
    parts = s_lower.split()
    if len(parts) < 2:
        return None

    qty_str, unit = parts[0], parts[1]
    if not qty_str.isdigit():
        return None

    qty = int(qty_str)
    if "hour" in unit:
        return now - timedelta(hours=qty)
    if "minute" in unit:
        return now - timedelta(minutes=qty)
    if "day" in unit:
        return now - timedelta(days=qty)
    return None


def parse_serpapi_date(date_str: str) -> datetime | None:
    """
    Parse SerpAPI Google News dates into timezone-aware UTC datetime.
    """
    if not date_str:
        return None

    s_clean = date_str.replace(",", "").strip()
    if not s_clean:
        return None

    s_lower = s_clean.lower()
    now = datetime.now(UTC)

    dt = _try_parse_absolute_date(s_clean)
    if dt is not None:
        return dt

    dt = _try_parse_absolute_date_with_time(s_clean)
    if dt is not None:
        return dt

    return _try_parse_relative(s_lower, now)
