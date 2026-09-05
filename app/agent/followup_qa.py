from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from app.agent.agent_prompts import (
    FOLLOWUP_ACTION_SYSTEM_PROMPT,
    FOLLOWUP_QA_SYSTEM_PROMPT,
    JOURNEY_ACTION_SYSTEM_PROMPT,
    JOURNEY_QA_SYSTEM_PROMPT,
)
from app.news.news_relevance import news_item_mentions_place, sanitize_answer_links, supports_high_impact_claim
from app.news.news_service import get_news_items, search_news
from app.routing.ors_service import plan_route
from app.travel_brief import build_travel_brief
from app.weather.weather_service import get_weather_line, get_weather_summary, get_weather_summary_by_coords

_LONG_DISTANCE_KM = 500.0


def _extract_text_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{4,}", (text or "").lower()))


def _match_news_item(
    question: str,
    last_reply: str | None,
    items: list[dict[str, Any]],
    place: str | None = None,
) -> dict[str, Any] | None:
    eligible_items = [
        item for item in items if isinstance(item, dict) and (not place or news_item_mentions_place(item, place))
    ]
    if not eligible_items:
        return None

    reference_text = " ".join(part for part in (question or "", last_reply or "") if part and part.strip()).strip()
    reference_tokens = _extract_text_tokens(reference_text)
    if not reference_tokens:
        return eligible_items[0]

    best_item = eligible_items[0]
    best_score = -1
    for item in eligible_items:
        item_tokens = _extract_text_tokens(_news_text(item))
        score = len(reference_tokens & item_tokens)
        if score > best_score:
            best_item = item
            best_score = score
    return best_item if best_score > 0 else None


def _news_text(item: dict[str, Any] | None) -> str:
    if not item:
        return ""

    return " ".join(
        part.strip()
        for part in (str(item.get("title") or ""), str(item.get("snippet") or ""))
        if part and str(part).strip()
    ).strip()


def _gather_place_evidence(place: str, horizon: str | None = None) -> dict[str, Any]:
    brief, err = build_travel_brief(place)
    weather_summary = brief.get("weather_summary")
    weather_reasons = brief.get("weather_reasons") or []
    weather_horizon = (horizon or "today").strip().lower()

    if weather_horizon not in ("today", "now"):
        summary, werr = get_weather_summary(place, weather_horizon)
        if summary and not werr:
            weather_summary = summary
            weather_reasons = []
    return {
        "place": place,
        "weather_summary": weather_summary,
        "weather_reasons": weather_reasons,
        "news_items": brief.get("news_items") or [],
        "news_reasons": brief.get("news_reasons") or [],
        "weather_horizon": weather_horizon,
        "source_error": err or None,
    }


async def _invoke_reasoner(llm: Any, system_prompt: str, *, place: str, question: str, evidence: dict[str, Any]) -> str:
    payload = json.dumps(evidence, ensure_ascii=True, indent=2)
    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Destination: {place}\nQuestion: {question}\nEvidence:\n{payload}",
            },
        ]
    )
    return str(getattr(response, "content", "") or "").strip()


async def _run_followup_reasoner(llm: Any, *, place: str, question: str, evidence: dict[str, Any]) -> str:
    return await _invoke_reasoner(
        llm,
        FOLLOWUP_QA_SYSTEM_PROMPT,
        place=place,
        question=question,
        evidence=evidence,
    )


async def _run_journey_reasoner(llm: Any, *, place: str, question: str, evidence: dict[str, Any]) -> str:
    return await _invoke_reasoner(
        llm,
        JOURNEY_QA_SYSTEM_PROMPT,
        place=place,
        question=question,
        evidence=evidence,
    )


async def _run_journey_transport_reasoner(llm: Any, *, place: str, question: str, evidence: dict[str, Any]) -> str:
    guidance_prompt = (
        f"{JOURNEY_QA_SYSTEM_PROMPT}\n\n"
        "Transport guidance:\n"
        "- If transport_guidance is present, follow it exactly.\n"
        "- Do not invent flight or ferry schedules.\n"
        "- If road travel is flagged as too long or infeasible, recommend checking flights/ferries.\n"
        "- If disruptions are flagged, mention them and suggest alternatives.\n"
        "- Keep the answer conversational and specific to origin/destination.\n"
    )
    return await _invoke_reasoner(
        llm,
        guidance_prompt,
        place=place,
        question=question,
        evidence=evidence,
    )


async def _plan_followup_action(llm: Any, *, place: str, question: str, evidence: dict[str, Any]) -> dict[str, Any]:
    raw = await _invoke_reasoner(
        llm,
        FOLLOWUP_ACTION_SYSTEM_PROMPT,
        place=place,
        question=question,
        evidence=evidence,
    )
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"answered": False, "answer": "", "search_query": ""}

    return {
        "answered": bool(parsed.get("answered")),
        "answer": str(parsed.get("answer") or "").strip(),
        "search_query": str(parsed.get("search_query") or "").strip(),
    }


async def _plan_journey_action(llm: Any, *, place: str, question: str, evidence: dict[str, Any]) -> dict[str, Any]:
    raw = await _invoke_reasoner(
        llm,
        JOURNEY_ACTION_SYSTEM_PROMPT,
        place=place,
        question=question,
        evidence=evidence,
    )
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"answered": False, "answer": "", "search_query": ""}

    return {
        "answered": bool(parsed.get("answered")),
        "answer": str(parsed.get("answer") or "").strip(),
        "search_query": str(parsed.get("search_query") or "").strip(),
    }


def _soften_followup_tone(final: str, place: str) -> str:
    text = (final or "").strip()
    if not text:
        return text

    replacements = (
        (
            re.compile(
                rf"^The retrieved news for {re.escape(place)} does not specify the answer to that question\.?$",
                re.IGNORECASE,
            ),
            f"I don't see anything in the current news for {place} that answers that directly.",
        ),
        (
            re.compile(
                rf"^The current weather data for {re.escape(place)} does not specify that detail\.?$",
                re.IGNORECASE,
            ),
            f"The current forecast for {place} doesn't spell that out.",
        ),
        (
            re.compile(r"^The retrieved reporting does not specify any possible disruptions\.?\s*", re.IGNORECASE),
            "I don't see any confirmed disruptions in the current reporting. ",
        ),
        (
            re.compile(r"^The retrieved reporting does not confirm that ", re.IGNORECASE),
            "I don't see anything in the current reporting that confirms ",
        ),
        (
            re.compile(r"^The retrieved reporting does not specify ", re.IGNORECASE),
            "I don't see anything in the current reporting that specifies ",
        ),
        (
            re.compile(r"^The retrieved weather data for ", re.IGNORECASE),
            "The current weather for ",
        ),
        (
            re.compile(r" does not specify any possible risks or weather disturbances\.?$", re.IGNORECASE),
            " doesn't point to any specific weather disruptions right now.",
        ),
        (
            re.compile(r" does not specify that detail\.?$", re.IGNORECASE),
            " doesn't spell that out.",
        ),
    )

    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)

    return re.sub(r"\s{2,}", " ", text).strip()


def _condense_direct_answer(final: str) -> str:
    text = (final or "").strip()
    if not text:
        return text

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(parts) <= 1:
        return text

    generic_patterns = (
        re.compile(r"\blooks generally fine for travel\b", re.IGNORECASE),
        re.compile(r"\blow risk level\b", re.IGNORECASE),
        re.compile(r"\brecent local reporting highlights\b", re.IGNORECASE),
        re.compile(r"\bcurrent news scan did not surface\b", re.IGNORECASE),
        re.compile(r"\brisk level\b", re.IGNORECASE),
    )
    answer_patterns = (
        re.compile(r"\b(?:not specified|doesn't say|does not say|doesn't confirm|does not confirm)\b", re.IGNORECASE),
        re.compile(r"\b(?:through|until|scheduled|expected|continues?|lasting|runs?)\b", re.IGNORECASE),
        re.compile(r"\b(?:yes|no)\b", re.IGNORECASE),
        re.compile(r"\b(?:i don't see|i can'?t|can'?t answer|cannot answer)\b", re.IGNORECASE),
    )

    non_generic = [part for part in parts if not any(pattern.search(part) for pattern in generic_patterns)]
    direct = [part for part in non_generic if any(pattern.search(part) for pattern in answer_patterns)]

    if direct:
        return direct[0].strip()
    if non_generic:
        return non_generic[0].strip()
    return parts[0].strip()


def _normalize_search_query(query: str, *extras: str) -> str:
    parts = [" ".join((query or "").split())]
    parts.extend(" ".join((extra or "").split()) for extra in extras if extra and extra.strip())
    return " ".join(part for part in parts if part).strip()


def _build_news_targeted_query(place: str, question: str, item: dict[str, Any] | None, _last_reply: str | None) -> str:
    if question and question.strip():
        return _normalize_search_query(question, place)
    if item and item.get("title"):
        return _normalize_search_query(str(item["title"]), place)
    return place


def _extract_best_news_link(evidence: dict[str, Any]) -> str | None:
    for key in ("matched_targeted_item", "matched_current_item"):
        link = _extract_link(evidence.get(key))
        if link:
            return link

    return None


def _has_matched_news_evidence(evidence: dict[str, Any]) -> bool:
    return any(isinstance(evidence.get(key), dict) for key in ("matched_targeted_item", "matched_current_item"))


def _validate_news_answer(final: str, evidence: dict[str, Any], place: str) -> str:
    allowed_links = {
        str(evidence[key].get("link") or "").strip()
        for key in ("matched_targeted_item", "matched_current_item")
        if isinstance(evidence.get(key), dict) and evidence[key].get("link")
    }
    final = sanitize_answer_links(final, allowed_links)
    if not final or not _has_matched_news_evidence(evidence):
        return final
    candidates = [
        evidence.get("matched_targeted_item"),
        evidence.get("matched_current_item"),
    ]
    if any(isinstance(item, dict) and supports_high_impact_claim(final, item, place) for item in candidates):
        return final
    return f"I couldn't confirm that specific update for {place} from the available news."


def _extract_link(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    link = str(item.get("link") or "").strip()
    return link or None


def _extract_link_from_items(items: Any) -> str | None:
    if not isinstance(items, list):
        return None
    for item in items:
        link = _extract_link(item)
        if link:
            return link
    return None


def _extract_link_from_block(block: Any) -> str | None:
    if not isinstance(block, dict):
        return None
    return _extract_link_from_items(block.get("news_items"))


def _answer_mentions_article_or_source(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in ("article", "source", "details", "read more", "here", "link"))


def _contains_url(text: str) -> bool:
    return bool(re.search(r"https?://\S+", text or ""))


def _append_followup_link_if_needed(final: str, evidence: dict[str, Any], original_text: str | None = None) -> str:
    source_text = original_text or final
    if not final or _contains_url(final) or not _answer_mentions_article_or_source(source_text):
        return final

    link = _extract_best_news_link(evidence)
    if not link:
        return final

    separator = "" if final.endswith((".", "!", "?")) else "."
    return f"{final}{separator} Source: {link}"


def _build_journey_targeted_query(origin: str, destination: str, question: str, pending_question: str | None) -> str:
    base = pending_question or question
    hazard_terms = "road closure traffic landslide flood ferry"
    return _normalize_search_query(base, origin, destination, hazard_terms)


def _detect_weather_horizon(question: str) -> str:
    lowered = (question or "").lower()
    for day in ("today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        if day in lowered:
            return day
    if "next week" in lowered:
        return "next week"
    return "today"


def _join_news_text(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if title:
            parts.append(title)
        if snippet:
            parts.append(snippet)
    return " ".join(parts).strip()


def _collect_disruption_flags(*blocks: dict[str, Any]) -> dict[str, bool]:
    items: list[dict[str, Any]] = []
    for block in blocks:
        block_items = block.get("news_items") if isinstance(block, dict) else None
        if isinstance(block_items, list):
            items.extend(block_items)

    text = _join_news_text(items).lower()
    if not text:
        return {"road": False, "flight": False, "ferry": False}

    road_terms = ("road closure", "road closed", "landslide", "bridge closure", "detour", "reroute", "re-route")
    flight_terms = (
        "flight cancellation",
        "flight cancelled",
        "flight canceled",
        "airport closure",
        "suspended flights",
    )
    ferry_terms = ("ferry cancellation", "ferry cancelled", "ferry canceled", "port closure", "ferry suspension")

    def _has_any(terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    return {"road": _has_any(road_terms), "flight": _has_any(flight_terms), "ferry": _has_any(ferry_terms)}


def _format_distance(distance_km: float | None) -> str:
    if not isinstance(distance_km, (int, float)):
        return ""
    return f"{distance_km:.0f} km"


def _format_duration(duration_min: float | None) -> str:
    if not isinstance(duration_min, (int, float)):
        return ""
    hours = duration_min / 60.0
    if hours >= 10:
        return f"about {hours:.0f} hours"
    return f"about {hours:.1f} hours"


def _guidance_no_routes() -> dict[str, Any]:
    return {
        "reason": "no_routes",
        "message": (
            "I couldn't find a drivable route between those locations. "
            "Please check ferry or flight schedules. As much as I want to provide you with the said schedules, "
            "I currently do not have access."
        ),
        "suggested_alternatives": ["flight", "ferry"],
    }


def _guidance_long_distance(
    *,
    distance_km: float,
    duration_min: float | None,
    road_disruption: bool,
) -> dict[str, Any]:
    distance_text = _format_distance(distance_km)
    duration_text = _format_duration(duration_min)
    distance_clause = f" ({distance_text}, {duration_text})" if distance_text and duration_text else ""
    detail = f"By road, this is a very long trip{distance_clause}."
    extra = " There are reports of road closures or disruptions that could affect driving." if road_disruption else ""
    return {
        "reason": "distance",
        "distance_km": distance_km,
        "duration_min": duration_min,
        "message": (
            f"{detail}{extra} Please check flight or ferry schedules. "
            "As much as I want to provide you with the said schedules, I currently do not have access."
        ),
        "suggested_alternatives": ["flight", "ferry"],
    }


def _guidance_road_disruption(
    *,
    mode_label: str,
    distance_km: float | None,
    duration_min: float | None,
) -> dict[str, Any]:
    distance_text = _format_distance(distance_km)
    duration_text = _format_duration(duration_min)
    distance_clause = f" ({distance_text}, {duration_text})" if distance_text and duration_text else ""
    return {
        "reason": "road_disruption",
        "distance_km": distance_km,
        "duration_min": duration_min,
        "message": (
            f"Driving by {mode_label} looks like the most practical option{distance_clause}, "
            "but there are reports of road closures or rerouting that could affect the trip. "
            "Please check flight or ferry options as a backup, since I don't have schedule access."
        ),
        "suggested_alternatives": ["flight", "ferry"],
    }


def _guidance_schedule_disruption(notes: list[str]) -> dict[str, Any]:
    summary = " and ".join(notes)
    return {
        "reason": "schedule_disruption",
        "message": (
            f"There are reports of {summary} for this route. "
            "Please confirm schedules directly; I don't have live schedule access."
        ),
        "suggested_alternatives": ["confirm_schedules"],
    }


def _build_transport_guidance(
    *,
    route_summary: dict[str, Any] | None,
    route_err: str | None,
    disruptions: dict[str, bool],
) -> dict[str, Any] | None:
    if route_err == "no_routes":
        return _guidance_no_routes()

    distance_km = route_summary.get("best_distance_km") if isinstance(route_summary, dict) else None
    duration_min = route_summary.get("best_duration_min") if isinstance(route_summary, dict) else None
    best_mode = route_summary.get("best_mode") if isinstance(route_summary, dict) else None

    if isinstance(distance_km, (int, float)) and distance_km > _LONG_DISTANCE_KM:
        return _guidance_long_distance(
            distance_km=distance_km,
            duration_min=duration_min,
            road_disruption=disruptions.get("road", False),
        )

    if disruptions.get("road") and best_mode:
        return _guidance_road_disruption(
            mode_label=str(best_mode),
            distance_km=distance_km,
            duration_min=duration_min,
        )

    notes: list[str] = []
    if disruptions.get("flight"):
        notes.append("flight disruptions")
    if disruptions.get("ferry"):
        notes.append("ferry disruptions")
    if notes:
        return _guidance_schedule_disruption(notes)

    return None


def _build_route_summary(route_plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not route_plan:
        return None
    routes = route_plan.get("routes") or []
    return {
        "best_mode": route_plan.get("best_mode"),
        "best_profile": route_plan.get("best_profile"),
        "best_distance_km": route_plan.get("best_distance_km"),
        "best_duration_min": route_plan.get("best_duration_min"),
        "modes": [
            {
                "mode": r.get("mode"),
                "distance_km": r.get("distance_km"),
                "duration_min": r.get("duration_min"),
            }
            for r in routes
            if isinstance(r, dict)
        ],
    }


def _build_midpoint_weather(route_plan: dict[str, Any] | None, horizon: str) -> dict[str, Any] | None:
    if not route_plan:
        return None
    midpoint = route_plan.get("midpoint") or {}
    lat = midpoint.get("lat")
    lon = midpoint.get("lon")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    summary, _mwerr = get_weather_summary_by_coords(
        float(lat),
        float(lon),
        horizon,
        label="En route midpoint",
    )
    return summary


def _get_transport_guidance(
    *,
    route_or_transport: bool,
    destination_evidence: dict[str, Any],
    origin_evidence: dict[str, Any],
    route_summary: dict[str, Any] | None,
    route_err: str | None,
) -> dict[str, Any] | None:
    if not route_or_transport:
        return None
    disruptions = _collect_disruption_flags(destination_evidence, origin_evidence)
    return _build_transport_guidance(
        route_summary=route_summary,
        route_err=route_err or None,
        disruptions=disruptions,
    )


async def _resolve_with_search(
    llm: Any,
    *,
    place: str,
    question: str,
    plan: dict[str, Any],
    base_evidence: dict[str, Any],
    build_query: Callable[[], str],
    run_reasoner: Callable[..., Any],
    search_fn: Callable[[str, str], tuple[list[dict[str, Any]], str]],
    enrich_evidence: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any], str]:
    if plan.get("answered") and plan.get("answer"):
        final = str(plan.get("answer") or "")
        return final, base_evidence, final

    targeted_query = str(plan.get("search_query") or "") or build_query()
    targeted_items, targeted_err = search_fn(targeted_query, place)
    evidence = {
        **base_evidence,
        "targeted_search_query": targeted_query,
        "targeted_search_error": targeted_err or None,
        "targeted_news_items": targeted_items[:3],
    }
    if enrich_evidence:
        evidence = enrich_evidence(evidence, targeted_items)

    final = await run_reasoner(llm, place=place, question=question, evidence=evidence)
    return final, evidence, final


async def answer_journey_question(
    llm: Any,
    place: str,
    question: str,
    origin: str,
    *,
    route_or_transport: bool,
    latest_user_message: str = "",
    conversation_history: list[dict[str, str]] | None = None,
    pending_question: str | None = None,
) -> dict[str, Any]:
    horizon = _detect_weather_horizon(pending_question or latest_user_message or question)
    destination_evidence = _gather_place_evidence(place, horizon)
    origin_evidence = _gather_place_evidence(origin, horizon)

    route_plan, route_err = plan_route(origin, place)
    route_summary = _build_route_summary(route_plan)
    midpoint_weather = _build_midpoint_weather(route_plan, horizon)

    current_evidence = {
        "mode": "journey_planning",
        "question": question,
        "latest_user_message": latest_user_message or question,
        "pending_question": pending_question,
        "conversation_history": conversation_history or [],
        "origin": origin,
        "destination": place,
        "route_or_transport": route_or_transport,
        "destination_evidence": destination_evidence,
        "origin_evidence": origin_evidence,
        "route_plan": route_plan,
        "route_summary": route_summary,
        "route_midpoint_weather": midpoint_weather,
        "route_plan_error": route_err or None,
        "targeted_search_query": None,
        "targeted_search_error": None,
        "targeted_news_items": [],
        "limitations": [
            "Routing data is based on OpenRouteService profiles and may not include live traffic or schedules.",
            "Public transit, flights, and ferry schedules are not available in this workflow.",
        ],
    }

    transport_guidance = _get_transport_guidance(
        route_or_transport=route_or_transport,
        destination_evidence=destination_evidence,
        origin_evidence=origin_evidence,
        route_summary=route_summary,
        route_err=route_err,
    )
    current_evidence["transport_guidance"] = transport_guidance
    plan = await _plan_journey_action(llm, place=place, question=question, evidence=current_evidence)
    final, evidence, raw_final = await _resolve_with_search(
        llm,
        place=place,
        question=question,
        plan=plan,
        base_evidence=current_evidence,
        build_query=lambda: _build_journey_targeted_query(origin, place, question, pending_question),
        run_reasoner=_run_journey_transport_reasoner if transport_guidance else _run_journey_reasoner,
        search_fn=search_news,
    )

    if not final:
        if route_or_transport:
            final = "I couldn't find a confirmed answer about the best transport from the data I gathered so far."
        else:
            final = "I couldn't find a confirmed answer from the data I gathered so far."

    final = _soften_followup_tone(final, place)
    final = _condense_direct_answer(final)
    final = _validate_news_answer(final, evidence, place)
    final = _append_followup_link_if_needed(final, evidence, raw_final)
    sources = [{"type": "weather"}, {"type": "news"}]
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}


async def answer_news_followup(
    llm: Any,
    place: str,
    question: str,
    last_reply: str | None,
    *,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    items, err = get_news_items(place)
    if err:
        final = f"I could not retrieve current news for {place} right now."
        return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": []}

    matched_item = _match_news_item(question, last_reply, items, place)
    current_evidence = {
        "mode": "news_followup",
        "question": question,
        "conversation_history": conversation_history or [],
        "current_news_items": items[:3],
        "matched_current_item": matched_item,
        "used_targeted_search": False,
        "targeted_search_query": None,
        "targeted_search_error": None,
        "targeted_news_items": [],
        "matched_targeted_item": None,
    }

    plan = await _plan_followup_action(llm, place=place, question=question, evidence=current_evidence)

    def _enrich_news_evidence(evidence: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            **evidence,
            "used_targeted_search": True,
            "matched_targeted_item": _match_news_item(question, last_reply, items, place),
        }

    final, evidence, raw_final = await _resolve_with_search(
        llm,
        place=place,
        question=question,
        plan=plan,
        base_evidence=current_evidence,
        build_query=lambda: _build_news_targeted_query(place, question, matched_item, last_reply),
        run_reasoner=_run_followup_reasoner,
        search_fn=search_news,
        enrich_evidence=_enrich_news_evidence,
    )
    if not final and not (plan.get("answered") and plan.get("answer")):
        final = f"I couldn't find a confirmed answer in the current news for {place}."

    final = _soften_followup_tone(final, place)
    final = _condense_direct_answer(final)
    final = _validate_news_answer(final, evidence, place)
    final = _append_followup_link_if_needed(final, evidence, raw_final)
    sources = [{"type": "news"}] if _has_matched_news_evidence(evidence) else []
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}


async def answer_general_followup(
    llm: Any,
    place: str,
    question: str,
    last_reply: str | None,
    *,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    place_evidence = _gather_place_evidence(place)
    matched_item = _match_news_item(question, last_reply, list(place_evidence.get("news_items") or []), place)
    current_evidence = {
        "mode": "general_followup",
        "question": question,
        "conversation_history": conversation_history or [],
        "place_evidence": place_evidence,
        "matched_current_item": matched_item,
        "used_targeted_search": False,
        "targeted_search_query": None,
        "targeted_search_error": None,
        "targeted_news_items": [],
        "matched_targeted_item": None,
    }

    plan = await _plan_followup_action(llm, place=place, question=question, evidence=current_evidence)

    def _enrich_general_evidence(evidence: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            **evidence,
            "used_targeted_search": True,
            "matched_targeted_item": _match_news_item(question, last_reply, items, place),
        }

    final, evidence, raw_final = await _resolve_with_search(
        llm,
        place=place,
        question=question,
        plan=plan,
        base_evidence=current_evidence,
        build_query=lambda: _build_news_targeted_query(place, question, matched_item, last_reply),
        run_reasoner=_run_followup_reasoner,
        search_fn=search_news,
        enrich_evidence=_enrich_general_evidence,
    )
    if not final and not (plan.get("answered") and plan.get("answer")):
        final = f"I couldn't find a confirmed answer from the data I gathered so far for {place}."

    final = _soften_followup_tone(final, place)
    final = _condense_direct_answer(final)
    final = _validate_news_answer(final, evidence, place)
    final = _append_followup_link_if_needed(final, evidence, raw_final)
    return {
        "place": place,
        "final": final,
        "risk_level": None,
        "travel_advice": [],
        "sources": [{"type": "weather"}, {"type": "news"}],
    }


async def answer_weather_followup(
    llm: Any,
    place: str,
    question: str,
    *,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    horizon = _detect_weather_horizon(question)
    summary, _ = get_weather_summary(place, horizon)
    sources = [{"type": "weather"}]

    if not summary:
        line, _ = get_weather_line(place)
        if line:
            evidence = {
                "mode": "weather_followup",
                "question": question,
                "conversation_history": conversation_history or [],
                "requested_horizon": horizon,
                "weather_snapshot": line,
                "weather_summary": None,
            }
            final = await _run_followup_reasoner(llm, place=place, question=question, evidence=evidence)
            if not final:
                final = f"The current weather snapshot for {place} is {line}."

            final = _soften_followup_tone(final, place)
            final = _condense_direct_answer(final)
            return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}

        final = f"I could not retrieve current weather data for {place} right now."
        return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": []}

    evidence = {
        "mode": "weather_followup",
        "question": question,
        "conversation_history": conversation_history or [],
        "requested_horizon": horizon,
        "weather_summary": summary,
        "weather_snapshot": None,
    }
    final = await _run_followup_reasoner(llm, place=place, question=question, evidence=evidence)
    if not final:
        final = f"I couldn't find a confirmed weather answer for {place} from the data I gathered so far."

    final = _soften_followup_tone(final, place)
    final = _condense_direct_answer(final)
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": sources}
