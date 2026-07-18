from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.news.news_service import get_news_items
from app.travel_concern_assessor import assess_travel_concern
from app.weather.weather_service import get_weather_line, get_weather_summary

RiskLevel = Literal["low", "medium", "high"]
SourceType = Literal["weather", "news"]


class BriefSource(TypedDict):
    type: SourceType


class TravelBrief(TypedDict):
    place: str
    final: str
    risk_level: RiskLevel
    travel_advice: list[str]
    sources: list[BriefSource]
    weather_summary: dict[str, Any] | None
    weather_reasons: list[str]
    news_reasons: list[str]
    news_items: list[dict[str, Any]]


def _build_sources(
    weather_summary: dict[str, Any] | None, weather_line: str, headlines: list[dict[str, Any]]
) -> list[BriefSource]:
    sources: list[BriefSource] = []
    if weather_summary or weather_line:
        sources.append({"type": "weather"})
    if headlines:
        sources.append({"type": "news"})
    return sources


def build_travel_brief(place: str) -> tuple[TravelBrief, str]:
    weather_summary, weather_summary_err = get_weather_summary(place, "today")
    weather_line = ""
    weather_line_err = ""
    if not weather_summary:
        weather_line, weather_line_err = get_weather_line(place)

    headlines, news_err = get_news_items(place)
    assessment = assess_travel_concern(
        place,
        weather_summary,
        weather_line,
        headlines,
        news_scan_available=not bool(news_err),
    )

    sources = _build_sources(weather_summary, weather_line, headlines)
    errors = [err for err in (weather_summary_err or weather_line_err, news_err) if err]

    brief: TravelBrief = {
        "place": place,
        "final": assessment["final"],
        "risk_level": assessment["risk_level"],
        "travel_advice": assessment["travel_advice"][:3],
        "sources": sources,
        "weather_summary": weather_summary,
        "weather_reasons": assessment["weather_reasons"],
        "news_reasons": assessment["news_reasons"],
        "news_items": assessment["relevant_news_items"][:3],
    }

    if not sources:
        brief["travel_advice"] = ["Retry shortly because current weather and news sources were unavailable"]

    return brief, "; ".join(errors)
