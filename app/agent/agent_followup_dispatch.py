from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.agent_policy import AnswerMode, needs_followup_reference_clarification, needs_origin_clarification


async def handle_pre_agent_paths(
    *,
    session_id: str,
    place: str,
    question: str | None,
    last_reply: str | None,
    recent_turns: list[dict[str, str]],
    answer_mode: AnswerMode,
    same_destination_followup: bool,
    effective_question: str | None,
    pending_question: str | None,
    origin: str | None,
    route_or_transport: bool,
    debug: bool,
    llm: Any,
    answer_news_fn: Callable[..., Any],
    answer_weather_fn: Callable[..., Any],
    answer_general_fn: Callable[..., Any],
    answer_journey_fn: Callable[..., Any],
    mark_tools_called_fn: Callable[..., Any],
    set_pending_agent_context_fn: Callable[..., Any],
    set_pending_journey_question_fn: Callable[..., Any],
    finalize_result_fn: Callable[..., Any],
) -> dict[str, Any] | None:
    if needs_followup_reference_clarification(question, last_reply):
        clarification = "I need the specific news item or the previous message to answer that follow-up directly."
        await mark_tools_called_fn(session_id, tool_names=[], user_message=question, agent_reply=clarification)
        result: dict[str, Any] = {
            "place": place,
            "final": clarification,
            "risk_level": None,
            "travel_advice": [],
            "sources": [],
        }
        return await finalize_result_fn(
            session_id=session_id, place=place, question=question, result=result, debug=debug
        )
    if answer_mode == "news_followup":
        result = await answer_news_fn(llm, place, question or "", last_reply, conversation_history=recent_turns)
        return await finalize_result_fn(
            session_id=session_id, place=place, question=question, result=result, debug=debug
        )
    if answer_mode == "weather_followup":
        result = await answer_weather_fn(llm, place, question or "", conversation_history=recent_turns)
        return await finalize_result_fn(
            session_id=session_id, place=place, question=question, result=result, debug=debug
        )
    if same_destination_followup and answer_mode not in {"news_followup", "weather_followup", "journey_planning"}:
        result = await answer_general_fn(llm, place, question or "", last_reply, conversation_history=recent_turns)
        return await finalize_result_fn(
            session_id=session_id, place=place, question=question, result=result, debug=debug
        )
    if answer_mode != "journey_planning":
        return None
    if needs_origin_clarification(question, last_reply):
        clarification = (
            f"I can assess conditions in {place}, but I need your departure location to judge the trip itself. "
            "Where are you traveling from?"
        )
        await set_pending_agent_context_fn(
            session_id,
            {
                "mode": "journey_planning",
                "awaiting": "origin",
                "question": question or "",
                "destination": place,
            },
        )
        await set_pending_journey_question_fn(session_id, question)
        journey_result: dict[str, Any] = {
            "place": place,
            "final": clarification,
            "risk_level": None,
            "travel_advice": [],
            "sources": [],
        }
        return await finalize_result_fn(
            session_id=session_id, place=place, question=question, result=journey_result, debug=debug
        )
    if not origin:
        return None
    await set_pending_agent_context_fn(session_id, None)
    await set_pending_journey_question_fn(session_id, None)
    result = await answer_journey_fn(
        llm,
        place,
        effective_question or question or "",
        origin,
        route_or_transport=route_or_transport,
        latest_user_message=question or "",
        conversation_history=recent_turns,
        pending_question=pending_question,
    )
    return await finalize_result_fn(
        session_id=session_id, place=place, question=question, result=result, debug=debug
    )
