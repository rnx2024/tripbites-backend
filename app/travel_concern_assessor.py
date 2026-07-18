from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langchain_openai import ChatOpenAI
from openai import OpenAIError

from app.settings import settings
from app.travel_intelligence import classify_risk_level, score_weather_risk

RiskLevel = Literal["low", "medium", "high"]


class ConcernAssessment(TypedDict):
    risk_level: RiskLevel
    final: str
    travel_advice: list[str]
    weather_reasons: list[str]
    news_reasons: list[str]
    relevant_news_items: list[dict[str, Any]]


_MAX_NEWS_ITEMS = 8
_MAX_SELECTED_NEWS = 3

_ASSESSOR_SYSTEM_PROMPT = """
You are a travel concern assessor.

You receive current destination weather data and recent local news headlines/snippets.
Your job is to decide what is actually concerning for travel and produce a concise travel-focused assessment.

Rules:
- Return only JSON with this shape:
  {
    "risk_level": "low" | "medium" | "high",
    "final": "short travel-focused summary",
    "travel_advice": ["...", "..."],
    "weather_reasons": ["...", "..."],
    "news_reasons": ["...", "..."],
    "relevant_news_indexes": [0, 2]
  }
- Select at most 3 relevant news indexes.
- Base the answer only on the supplied evidence. Do not guess, overstate risk, or invent travel impact.
- Ignore news that is not meaningfully relevant to travelers.
- Do not treat generic civic, pricing, aid, research, policy, ordinance, or program stories as travel concerns unless the provided title or snippet clearly connects them to movement, access, schedules, safety, or crowding.
- If a news item's travel impact is unclear, exclude it.
- Keep "final" concise, natural, and conversational, but still factual and travel-focused.
- If the trip generally looks fine, say that plainly.
- If the evidence is thin or incomplete, say that plainly instead of filling gaps.
- "travel_advice" should contain only practical advice justified by the evidence.
"""

_llm = ChatOpenAI(
    model=settings.openrouter_model,
    temperature=0.0,
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
)


def _normalize_risk_level(value: Any) -> RiskLevel:
    normalized = str(value or "").strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized  # type: ignore[return-value]
    return "low"


def _normalize_text_list(value: Any, *, limit: int = 3) -> list[str]:
    if not isinstance(value, list):
        return []

    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text or text in items:
            continue
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _normalize_news_indexes(value: Any, total_items: int) -> list[int]:
    if not isinstance(value, list):
        return []

    indexes: list[int] = []
    for item in value:
        if not isinstance(item, int):
            continue
        if item < 0 or item >= total_items or item in indexes:
            continue
        indexes.append(item)
        if len(indexes) >= _MAX_SELECTED_NEWS:
            break
    return indexes


def _build_fallback_assessment(
    place: str,
    weather_summary: dict[str, Any] | None,
    weather_line: str,
    news_scan_available: bool,
) -> ConcernAssessment:
    weather_score, weather_reasons = score_weather_risk(weather_summary)
    risk_level = _normalize_risk_level(classify_risk_level(weather_score))

    if weather_summary:
        current = weather_summary.get("current") or {}
        day = weather_summary.get("day") or {}
        weather_text = str(current.get("weather_text") or "").strip().lower()
        tmin = day.get("tmin_c")
        tmax = day.get("tmax_c")
        temp_part = f" around {tmin}-{tmax}°C" if tmin is not None and tmax is not None else ""
        final = f"{place} looks generally manageable for travel based on the current weather"
        if weather_text:
            final += f", with {weather_text} conditions{temp_part}."
        else:
            final += "."
    elif weather_line:
        final = f"{place} travel conditions were summarized from the latest weather snapshot: {weather_line}."
    elif news_scan_available:
        final = f"{place} travel conditions look generally manageable from the currently gathered data."
    else:
        final = f"{place} travel conditions could not be fully assessed from the currently gathered data."

    advice: list[str] = []
    if not news_scan_available:
        advice.append("Local news context could not be confirmed from the current scan.")

    return {
        "risk_level": risk_level,
        "final": final,
        "travel_advice": advice,
        "weather_reasons": weather_reasons,
        "news_reasons": [],
        "relevant_news_items": [],
    }


def assess_travel_concern(
    place: str,
    weather_summary: dict[str, Any] | None,
    weather_line: str,
    headlines: list[dict[str, Any]],
    *,
    news_scan_available: bool,
) -> ConcernAssessment:
    news_payload = [
        {
            "index": idx,
            "title": str(item.get("title") or "").strip(),
            "snippet": str(item.get("snippet") or "").strip(),
            "source": str(item.get("source") or "").strip(),
            "date": str(item.get("date") or "").strip(),
            "link": str(item.get("link") or "").strip(),
        }
        for idx, item in enumerate(headlines[:_MAX_NEWS_ITEMS])
    ]

    evidence = {
        "destination": place,
        "weather_summary": weather_summary,
        "weather_line": weather_line or None,
        "news_scan_available": news_scan_available,
        "news_items": news_payload,
    }

    try:
        response = _llm.invoke(
            [
                {"role": "system", "content": _ASSESSOR_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=True, indent=2)},
            ]
        )
        parsed = json.loads(str(getattr(response, "content", "") or "").strip())
    except (OpenAIError, ValueError, TypeError):
        return _build_fallback_assessment(place, weather_summary, weather_line, news_scan_available)

    selected_indexes = _normalize_news_indexes(parsed.get("relevant_news_indexes"), len(news_payload))
    relevant_news_items = [headlines[idx] for idx in selected_indexes]
    risk_level = _normalize_risk_level(parsed.get("risk_level"))
    final = str(parsed.get("final") or "").strip()

    if not final:
        return _build_fallback_assessment(place, weather_summary, weather_line, news_scan_available)

    return {
        "risk_level": risk_level,
        "final": final,
        "travel_advice": _normalize_text_list(parsed.get("travel_advice")),
        "weather_reasons": _normalize_text_list(parsed.get("weather_reasons"), limit=4),
        "news_reasons": _normalize_text_list(parsed.get("news_reasons"), limit=4),
        "relevant_news_items": relevant_news_items,
    }
