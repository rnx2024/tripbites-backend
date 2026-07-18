# app/agent_tools.py (UPDATED: delegates helpers; public tool names/signatures unchanged)
from __future__ import annotations

import json
from typing import Any, Optional
from pydantic import BaseModel
from langchain_core.tools import tool

from app.weather.weather_service import get_weather_line, get_weather_summary
from app.news.news_service import get_news_items, search_news
from app.travel_brief import build_travel_brief
from app.routing.ors_service import plan_route

from app.tooling.sync_cache import (
    CACHE_TTL_SECONDS_DEFAULT,
    cache_get_json,
    cache_get_str,
    cache_set_json,
    cache_set_str,
)
from app.tooling.text_normalize import normalize_text
from app.tooling.retry_rate_limit import ERROR_PREFIX, RateLimiter, is_error_result


# -----------------------------
# Global cache (Redis) for tools
# -----------------------------
CACHE_TTL_SECONDS = CACHE_TTL_SECONDS_DEFAULT

# -----------------------------
# Initialize rate limiters
# -----------------------------
weather_rate = RateLimiter(5, 1.0)
news_rate = RateLimiter(2, 1.0)


# -----------------------------
# Tool Schemas
# -----------------------------
class WeatherInput(BaseModel):
    place: str
    horizon: Optional[str] = "today"


class NewsInput(BaseModel):
    place: str


class NewsSearchInput(BaseModel):
    query: str
    place_hint: Optional[str] = None


class RiskInput(BaseModel):
    place: str
    horizon: Optional[str] = "today"
    activity: Optional[str] = None


class TravelBriefInput(BaseModel):
    place: str


class RoutePlanInput(BaseModel):
    origin: str
    destination: str
    profiles: Optional[list[str]] = None


def _run(fn: Any) -> Any:
    """
    Execute a tool call once. HTTP-level retries (retryguard + tenacity) already
    handle transient upstream failures, so this only converts a raised exception
    into an "ERROR: ..." string for cache-skip checks via is_error_result().
    """
    try:
        return fn()
    except Exception as e:
        return f"{ERROR_PREFIX}{e}"


def _format_news_items(headlines: list[dict[str, Any]], *, empty_message: str) -> str:
    if not headlines:
        return empty_message

    return "\n".join(
        f"- {h['title']} ({h['source']}, {h['date']})"
        + (f" — {h['snippet'][:160].strip()}" if h.get("snippet") else "")
        + (f" -> {h['link']}" if h.get("link") else "")
        for h in headlines
    )


def _weather_line_or_raise(place: str) -> str:
    line, err = get_weather_line(place)
    if err:
        raise RuntimeError(err)
    return line or "No weather data."


def _load_weather_summary(place: str, horizon: str) -> dict[str, Any]:
    summary, err = get_weather_summary(place, horizon)
    if err or not summary:
        raise RuntimeError(err or "No forecast data.")
    return summary


def _format_weather_summary(summary: dict[str, Any], horizon: str, place: str) -> str:
    cur = summary.get("current") or {}
    day = summary.get("day") or {}
    place_label = summary.get("place_label") or place
    label = day.get("label") or horizon
    wx = cur.get("weather_text") or "n/a"
    parts = [f"{place_label} ({label}): {wx}"]
    tmin = day.get("tmin_c")
    tmax = day.get("tmax_c")
    if tmin is not None and tmax is not None:
        parts.append(f"{tmin}–{tmax}°C")
    precip = day.get("precip_mm")
    if precip is not None:
        parts.append(f"precip ~{precip} mm")
    wind = day.get("wind_speed_max_kmh")
    if wind is not None:
        parts.append(f"wind max ~{wind} km/h")
    return ", ".join(parts)


def _load_cached_weather_summary(place: str, horizon: str) -> Any:
    w_key = f"cache:tool:weather_summary:{normalize_text(place)}:{normalize_text(horizon)}"
    cached = cache_get_json(w_key)
    if cached is not None:
        return cached

    summary, werr = get_weather_summary(place, horizon)
    if werr and not summary:
        raise RuntimeError(werr)
    cache_set_json(w_key, summary, ttl=CACHE_TTL_SECONDS)
    return summary


def _load_cached_news_items(place: str) -> list[dict[str, Any]]:
    n_key = f"cache:tool:news_items:{normalize_text(place)}"
    cached = cache_get_json(n_key)
    if cached is not None:
        return cached

    headlines, _ = get_news_items(place)
    cache_set_json(n_key, headlines or [], ttl=CACHE_TTL_SECONDS)
    return headlines or []


def _extract_risk_reasons(brief: dict[str, Any]) -> list[str]:
    reasons = list(dict.fromkeys((brief.get("weather_reasons") or []) + (brief.get("news_reasons") or [])))
    if not reasons:
        reasons = list(brief.get("travel_advice") or [])
    return reasons


def _build_risk_message(level: str, reasons: list[str], activity: Optional[str]) -> str:
    msg = f"Risk level: {level}. "
    if reasons:
        msg += "Key factors: " + "; ".join(dict.fromkeys(reasons)) + "."
    if activity:
        msg += f" Activity: {activity}."
    return msg


# -----------------------------
# Tools
# -----------------------------
@tool(args_schema=TravelBriefInput)
def travel_brief_tool(place: str) -> str:
    """Build a structured travel brief for a destination using current weather and local news signals."""
    cache_key = f"cache:tool:travel_brief:{normalize_text(place)}"
    cached = cache_get_str(cache_key)
    if cached is not None:
        return cached

    weather_rate.acquire()
    news_rate.acquire()

    def call() -> str:
        brief, err = build_travel_brief(place)
        if err and not brief["sources"]:
            raise RuntimeError(err)
        return json.dumps(brief)

    out = _run(call)
    if isinstance(out, str) and not is_error_result(out):
        cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
    return out


@tool(args_schema=WeatherInput)
def weather_tool(place: str, horizon: Optional[str] = "today") -> str:
    """Get a concise weather summary for a specific place and time horizon."""
    hz = (horizon or "today").strip().lower()
    cache_key = f"cache:tool:weather_line:{normalize_text(place)}:{normalize_text(hz)}"
    cached = cache_get_str(cache_key)
    if cached is not None:
        return cached

    weather_rate.acquire()

    def call():
        if hz in ("today", "now"):
            return _weather_line_or_raise(place)

        summary = _load_weather_summary(place, hz)
        return _format_weather_summary(summary, hz, place)

    out = _run(call)
    if isinstance(out, str) and not is_error_result(out):
        cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
    return out


@tool(args_schema=NewsInput)
def news_tool(place: str) -> str:
    """Fetch recent news headlines for a specific city or region."""
    cache_key = f"cache:tool:news_lines:{normalize_text(place)}"
    cached = cache_get_str(cache_key)
    if cached is not None:
        return cached

    news_rate.acquire()

    def call():
        headlines, err = get_news_items(place)
        if err:
            raise RuntimeError(err)
        return _format_news_items(headlines, empty_message="No recent news.")

    out = _run(call)
    if isinstance(out, str) and not is_error_result(out):
        cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
    return out


@tool(args_schema=NewsSearchInput)
def news_search_tool(query: str, place_hint: Optional[str] = None) -> str:
    """
    Run a targeted follow-up news search for a named issue and destination,
    such as 'PISTON strike Vigan', when the current snippets are insufficient.
    """
    q = (query or "").strip()
    hint = (place_hint or "").strip()
    if not q:
        raise RuntimeError("A search query is required.")

    cache_key = f"cache:tool:news_search:{normalize_text(q)}:{normalize_text(hint or 'global')}"
    cached = cache_get_str(cache_key)
    if cached is not None:
        return cached

    news_rate.acquire()

    def call():
        headlines, err = search_news(q, hint or None)
        if err:
            raise RuntimeError(err)
        return _format_news_items(headlines, empty_message="No targeted news results.")

    out = _run(call)
    if isinstance(out, str) and not is_error_result(out):
        cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
    return out


@tool(args_schema=RiskInput)
def city_risk_tool(
    place: str,
    horizon: str = "today",
    activity: Optional[str] = None,
) -> str:
    """
    Assess city risk level (LOW, MEDIUM, HIGH) for outdoor activity and also consider travel conditions
    based on forecasted weather and recent local news.
    """
    def call() -> str:
        weather_rate.acquire()
        news_rate.acquire()

        _load_cached_weather_summary(place, horizon)
        _load_cached_news_items(place)
        brief, err = build_travel_brief(place)
        if err and not brief["sources"]:
            raise RuntimeError(err)

        level = str(brief.get("risk_level") or "low").upper()
        reasons = _extract_risk_reasons(brief)
        return _build_risk_message(level, reasons, activity)

    return _run(call)


@tool(args_schema=RoutePlanInput)
def route_planner_tool(origin: str, destination: str, profiles: Optional[list[str]] = None) -> str:
    """Plan a route between two locations using OpenRouteService and return distance/duration by mode."""
    origin_norm = normalize_text(origin) or "unknown"
    dest_norm = normalize_text(destination) or "unknown"
    prof_key = ",".join(profiles or [])
    cache_key = f"cache:tool:route_plan:{origin_norm}:{dest_norm}:{normalize_text(prof_key) or 'default'}"
    cached = cache_get_str(cache_key)
    if cached is not None:
        return cached

    def call() -> str:
        plan, err = plan_route(origin, destination, tuple(profiles) if profiles else None)
        if err or not plan:
            raise RuntimeError(err or "route_planning_failed")
        return json.dumps(plan)

    out = _run(call)
    if isinstance(out, str) and not is_error_result(out):
        cache_set_str(cache_key, out, ttl=CACHE_TTL_SECONDS)
    return out
