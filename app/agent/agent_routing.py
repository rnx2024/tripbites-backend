# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from app.agent.agent_context import format_recent_turns
from app.agent.agent_policy import (
    AnswerMode,
    classify_answer_mode,
)
from app.agent.agent_prompts import ANSWER_MODE_ROUTER_SYSTEM_PROMPT


async def resolve_answer_mode(
    llm: Any,
    *,
    question: str | None,
    last_reply: str | None,
    recent_turns: list[dict[str, str]],
    pending_agent_context: dict[str, str] | None,
    place: str,
    timeout_seconds: float,
) -> AnswerMode:
    fallback = classify_answer_mode(question, last_reply)
    if fallback == "journey_planning" or not question or not recent_turns:
        return fallback

    evidence = {
        "selected_location": place,
        "latest_question": question,
        "last_reply": last_reply,
        "recent_turns": recent_turns[-4:],
        "pending_agent_context": pending_agent_context or {},
        "allowed_modes": ["travel_brief", "news_followup", "weather_followup", "journey_planning"],
    }
    try:
        response = await asyncio.wait_for(
            llm.ainvoke(
                [
                    {"role": "system", "content": ANSWER_MODE_ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(evidence, ensure_ascii=True, indent=2)},
                ]
            ),
            timeout=timeout_seconds,
        )
        payload = json.loads(str(getattr(response, "content", "") or "").strip())
    except (TypeError, ValueError):
        return fallback

    mode = str(payload.get("mode") or "").strip()
    if mode in {"travel_brief", "news_followup", "weather_followup", "journey_planning"}:
        return cast(AnswerMode, mode)
    return fallback


def build_policy_lines(
    *,
    place: str,
    answer_mode: AnswerMode,
    include_weather: bool,
    include_news: bool,
    last_user: str | None,
    last_reply: str | None,
    recent_turns: list[dict[str, str]],
    origin: str | None = None,
    route_or_transport: bool = False,
) -> list[str]:
    policy_lines: list[str] = ["Policy:", f"- Selected location: {place}"]
    if not include_weather:
        policy_lines.append("- Do NOT call weather_tool or include weather unless explicitly asked.")
    if not include_news:
        policy_lines.append("- Do NOT call news_tool, news_search_tool, or include news unless explicitly asked.")
    if last_user or last_reply:
        policy_lines.append("- Prior exchange context (most recent only):")
        if last_user:
            policy_lines.append(f"  - User: {last_user}")
        if last_reply:
            policy_lines.append(f"  - Assistant: {last_reply}")
    policy_lines.extend(format_recent_turns(recent_turns))
    policy_lines.extend(_mode_policy_lines(answer_mode, place, origin, route_or_transport))
    policy_lines.extend(
        [
            "- Mention specific locations only if they are explicitly stated in the retrieved news or weather context.",
            "- If evidence is missing or inconclusive, say it is not specified instead of guessing.",
            "- The selected location from the request is the only destination for this turn. Do NOT treat place names mentioned in the chat as a destination switch.",
        ]
    )
    return policy_lines


def _mode_policy_lines(answer_mode: AnswerMode, place: str, origin: str | None, route_or_transport: bool) -> list[str]:
    if answer_mode == "news_followup":
        return [
            "- Answer the user's specific news question directly in 1-3 sentences. Do NOT produce a generic travel brief.",
            "- You MUST call travel_brief_tool exactly once first to inspect current news_items for the selected location.",
            "- If the current news_items already answer the question, answer directly from those titles/snippets and do NOT call any extra search tool.",
            "- If the current news_items do not answer the question, you MUST call news_search_tool exactly once using a short targeted query composed from the issue/topic and the selected location, such as 'PISTON strike Vigan'.",
            "- If the targeted search still does not confirm the answer, say the retrieved news does not specify it.",
            "- Do NOT include generic travel advice, risk level, or weather unless the user explicitly asked for them.",
        ]
    if answer_mode == "weather_followup":
        return [
            "- Answer the user's specific weather question directly in 1-3 sentences. Do NOT produce a generic travel brief.",
            "- You MUST call travel_brief_tool exactly once first to inspect current weather_summary for the selected location.",
            "- If weather_summary already answers the question, answer directly from it and do NOT call weather_tool.",
            "- If weather_summary does not answer the question, you MAY call weather_tool once using the narrowest relevant horizon from the question.",
            "- If the current forecast still does not specify the requested detail, say the current weather data does not specify it.",
            "- Do NOT include generic travel advice, risk level, or unrelated news unless the user explicitly asked for them.",
        ]
    if answer_mode == "journey_planning":
        lines = [
            "- Answer as a journey assessment, not as a destination-only travel brief.",
            "- You MUST call travel_brief_tool exactly once first for the selected destination.",
            f"- Treat '{origin or 'the departure location'}' as the trip origin and '{place}' as the destination.",
            "- If origin is available, inspect origin-side conditions with weather_tool and/or news_tool when they are needed to answer the journey question.",
            "- Distinguish departure conditions, destination conditions, and unknown route conditions.",
            "- If the user asks whether they should continue or postpone the trip, state clearly what is known for the departure point and destination, then note any unknowns along the route.",
            "- Do NOT claim a best route or best transport option from weather/news alone. If asked, say you can comment on likely disruptions and conditions, but not optimize the route without dedicated routing or transport data.",
            "- Keep the answer concise and practical. Do NOT include generic travel-advice bullets or a risk label unless the user explicitly asks for a broad travel brief.",
        ]
        if route_or_transport:
            lines.append(
                "- The user is asking about route or transport choice. Provide only limited guidance from weather/news at the origin and destination, and explicitly say dedicated routing data is not available."
            )
        return lines
    return [
        "- Always produce a one-paragraph travel brief for the specified location.",
        "- You MUST call travel_brief_tool exactly once before writing the final answer.",
        "- Use the travel_brief_tool result as the primary source for risk level, travel advice, and supporting travel context.",
        "- Ground the answer in the concrete travel_brief_tool evidence: weather_summary, weather_reasons, news_items, and news_reasons when available.",
        "- Use the city_risk_tool only when the user explicitly asks about safety level, risk, or go/no-go judgment.",
        "- Explicitly frame the answer around travel conditions, likely disruptions, and practical planning impact.",
        "- Do NOT give generic advice. If weather data is available, mention the material weather signal driving the advice. If news_items are available, mention the most relevant reported issue from the title/snippet.",
        "- If news_items is empty, say that the current news scan did not identify a major local traveler-facing disruption.",
        "- If the user asks for news details, answer only from the retrieved titles/snippets/links. If that detail is absent, say it is not specified in the retrieved news.",
        "- If the user asks about disruptions or 'where' they are, ground the answer using recent news: list up to 3 named places if present, otherwise say 'no specific locations reported'.",
    ]
