from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agent.agent_policy import AnswerMode, needs_followup_reference_clarification, needs_origin_clarification


@dataclass(frozen=True)
class FollowupRequest:
    session_id: str
    place: str
    question: str | None
    last_reply: str | None
    recent_turns: list[dict[str, str]]
    answer_mode: AnswerMode
    same_destination_followup: bool
    effective_question: str | None
    pending_question: str | None
    origin: str | None
    route_or_transport: bool
    debug: bool


@dataclass(frozen=True)
class FollowupDispatchDependencies:
    llm: Any
    answer_news: Callable[..., Any]
    answer_weather: Callable[..., Any]
    answer_general: Callable[..., Any]
    answer_journey: Callable[..., Any]
    mark_tools_called: Callable[..., Any]
    set_pending_agent_context: Callable[..., Any]
    set_pending_journey_question: Callable[..., Any]
    finalize_result: Callable[..., Any]


async def handle_pre_agent_paths(
    request: FollowupRequest, dependencies: FollowupDispatchDependencies
) -> dict[str, Any] | None:
    if needs_followup_reference_clarification(request.question, request.last_reply):
        return await _handle_reference_clarification(request, dependencies)
    result = await _handle_standard_followup(request, dependencies)
    if result is not None:
        return result
    return await _handle_journey_followup(request, dependencies)


async def _handle_reference_clarification(
    request: FollowupRequest, dependencies: FollowupDispatchDependencies
) -> dict[str, Any]:
    clarification = "I need the specific news item or the previous message to answer that follow-up directly."
    await dependencies.mark_tools_called(
        request.session_id,
        tool_names=[],
        user_message=request.question,
        agent_reply=clarification,
    )
    result = _empty_result(request.place, clarification)
    return await _finalize(request, dependencies, result)


async def _handle_standard_followup(
    request: FollowupRequest, dependencies: FollowupDispatchDependencies
) -> dict[str, Any] | None:
    if request.answer_mode == "news_followup":
        result = await dependencies.answer_news(
            dependencies.llm,
            request.place,
            request.question or "",
            request.last_reply,
            conversation_history=request.recent_turns,
        )
        return await _finalize(request, dependencies, result)
    if request.answer_mode == "weather_followup":
        result = await dependencies.answer_weather(
            dependencies.llm,
            request.place,
            request.question or "",
            conversation_history=request.recent_turns,
        )
        return await _finalize(request, dependencies, result)
    if request.same_destination_followup and request.answer_mode == "travel_brief":
        result = await dependencies.answer_general(
            dependencies.llm,
            request.place,
            request.question or "",
            request.last_reply,
            conversation_history=request.recent_turns,
        )
        return await _finalize(request, dependencies, result)
    return None


async def _handle_journey_followup(
    request: FollowupRequest, dependencies: FollowupDispatchDependencies
) -> dict[str, Any] | None:
    if request.answer_mode != "journey_planning":
        return None
    if needs_origin_clarification(request.question, request.last_reply):
        return await _request_origin(request, dependencies)
    if not request.origin:
        return None
    await dependencies.set_pending_agent_context(request.session_id, None)
    await dependencies.set_pending_journey_question(request.session_id, None)
    result = await dependencies.answer_journey(
        dependencies.llm,
        request.place,
        request.effective_question or request.question or "",
        request.origin,
        route_or_transport=request.route_or_transport,
        latest_user_message=request.question or "",
        conversation_history=request.recent_turns,
        pending_question=request.pending_question,
    )
    return await _finalize(request, dependencies, result)


async def _request_origin(
    request: FollowupRequest, dependencies: FollowupDispatchDependencies
) -> dict[str, Any]:
    clarification = (
        f"I can assess conditions in {request.place}, but I need your departure location to judge the trip itself. "
        "Where are you traveling from?"
    )
    await dependencies.set_pending_agent_context(
        request.session_id,
        {
            "mode": "journey_planning",
            "awaiting": "origin",
            "question": request.question or "",
            "destination": request.place,
        },
    )
    await dependencies.set_pending_journey_question(request.session_id, request.question)
    return await _finalize(request, dependencies, _empty_result(request.place, clarification))


def _empty_result(place: str, final: str) -> dict[str, Any]:
    return {"place": place, "final": final, "risk_level": None, "travel_advice": [], "sources": []}


async def _finalize(
    request: FollowupRequest, dependencies: FollowupDispatchDependencies, result: dict[str, Any]
) -> dict[str, Any]:
    return await dependencies.finalize_result(
        session_id=request.session_id,
        place=request.place,
        question=request.question,
        result=result,
        debug=request.debug,
    )
