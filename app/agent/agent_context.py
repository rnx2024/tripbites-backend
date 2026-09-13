from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.agent.agent_policy import (
    AnswerMode,
    asks_route_or_transport,
    extract_origin,
    is_journey_planning_question,
    is_origin_only_reply,
)
from app.session.session_cache import (
    set_pending_agent_context,
    set_pending_journey_question,
)


def build_user_prompt(place: str, question: str | None, origin: str | None = None) -> str:
    if not question:
        return (
            "Provide a concise travel brief for the destination below. Focus on travel conditions, likely disruptions, "
            f"and what matters most for someone going there today: {place}."
        )
    parts = [f"Location: {place}\nQuestion: {question}\n"]
    if origin:
        parts.append(f"Journey origin: {origin}\n")
    parts.append("Answer as ONE concise travel-oriented paragraph, plain text.")
    return "".join(parts)


def format_recent_turns(recent_turns: list[dict[str, str]]) -> list[str]:
    if not recent_turns:
        return []
    lines = ["- Recent conversation context:"]
    for turn in recent_turns[-4:]:
        user_text = str(turn.get("user") or "").strip()
        assistant_text = str(turn.get("assistant") or "").strip()
        if user_text:
            lines.append(f"  - User: {user_text}")
        if assistant_text:
            lines.append(f"  - Assistant: {assistant_text}")
    return lines


def has_same_destination_followup(
    *,
    question: str | None,
    place: str,
    active_destination: str | None,
    last_reply: str | None,
    recent_turns: list[dict[str, str]],
    pending_agent_context: dict[str, str] | None,
    pending_journey_question: str | None,
) -> bool:
    if not question or active_destination != place:
        return False
    return bool(pending_agent_context or pending_journey_question or recent_turns or (last_reply or "").strip())


async def reset_session_for_destination_change(
    *,
    session_id: str,
    place: str,
    active_destination: str | None,
    recent_turns: list[dict[str, str]],
    pending_agent_context: dict[str, str] | None,
    pending_journey_question: str | None,
) -> tuple[list[dict[str, str]], dict[str, str] | None, str | None]:
    if active_destination and active_destination != place:
        await set_pending_agent_context(session_id, None)
        await set_pending_journey_question(session_id, None)
        return [], None, None
    return recent_turns, pending_agent_context, pending_journey_question


def _apply_active_origin(origin: str | None, active_origin: str | None, question: str | None) -> str | None:
    if origin:
        return origin
    if active_origin and question and (is_journey_planning_question(question) or asks_route_or_transport(question)):
        return active_origin
    return origin


def extract_request_origin(
    *, question: str | None, last_reply: str | None, active_origin: str | None
) -> str | None:
    return _apply_active_origin(extract_origin(question, last_reply), active_origin, question)


def is_awaiting_origin(pending_agent_context: dict[str, str] | None) -> bool:
    return (pending_agent_context or {}).get("awaiting") == "origin"


def restore_pending_journey_question(
    *,
    origin: str | None,
    question: str | None,
    last_reply: str | None,
    last_user: str | None,
    pending_question: str | None,
    pending_journey_question: str | None,
    awaiting_origin: bool,
) -> str | None:
    effective_question = question
    if awaiting_origin and origin:
        return pending_question or pending_journey_question or last_user or question
    if origin and "where are you traveling from" in (last_reply or "").lower() and not effective_question:
        return pending_journey_question or last_user or question
    return effective_question


def restore_origin_only_reply(
    *, origin: str | None, question: str | None, effective_question: str | None, last_user: str | None
) -> str | None:
    if (
        origin
        and effective_question == question
        and is_origin_only_reply(question)
        and last_user
        and (is_journey_planning_question(last_user) or asks_route_or_transport(last_user))
    ):
        return last_user
    return effective_question


def resolve_request_context(
    *,
    question: str | None,
    last_reply: str | None,
    last_user: str | None,
    pending_agent_context: dict[str, str] | None,
    pending_journey_question: str | None,
    active_origin: str | None,
) -> tuple[str | None, str | None, bool, str | None]:
    origin = extract_request_origin(question=question, last_reply=last_reply, active_origin=active_origin)
    pending_question = (pending_agent_context or {}).get("question")
    awaiting_origin = is_awaiting_origin(pending_agent_context)
    if awaiting_origin:
        origin = origin or extract_origin(question, "Where are you traveling from?")
    effective_question = restore_pending_journey_question(
        origin=origin,
        question=question,
        last_reply=last_reply,
        last_user=last_user,
        pending_question=pending_question,
        pending_journey_question=pending_journey_question,
        awaiting_origin=awaiting_origin,
    )
    effective_question = restore_origin_only_reply(
        origin=origin,
        question=question,
        effective_question=effective_question,
        last_user=last_user,
    )
    return origin, effective_question, awaiting_origin, pending_question


def finalize_origin(
    *, origin: str | None, effective_question: str | None, last_reply: str | None, active_origin: str | None
) -> str | None:
    resolved_origin = origin or extract_origin(effective_question, last_reply)
    return _apply_active_origin(resolved_origin, active_origin, effective_question)


@dataclass(frozen=True)
class PreparedRequest:
    question: str | None
    effective_question: str | None
    origin: str | None
    answer_mode: AnswerMode
    route_or_transport: bool
    last_user: str | None
    last_reply: str | None
    recent_turns: list[dict[str, str]]
    same_destination_followup: bool
    pending_question: str | None


async def prepare_request_context(
    *,
    session_id: str,
    place: str,
    question: str | None,
    answer_mode_resolver: Callable[..., Awaitable[AnswerMode]],
    get_last_exchange_fn: Callable[..., Awaitable[tuple[str | None, str | None]]],
    get_recent_turns_fn: Callable[..., Awaitable[list[dict[str, str]]]],
    get_active_destination_fn: Callable[..., Awaitable[str | None]],
    get_active_origin_fn: Callable[..., Awaitable[str | None]],
    get_pending_agent_context_fn: Callable[..., Awaitable[dict[str, str] | None]],
    get_pending_journey_question_fn: Callable[..., Awaitable[str | None]],
    set_active_origin_fn: Callable[..., Awaitable[None]],
) -> PreparedRequest:
    last_user, last_reply = await get_last_exchange_fn(session_id)
    recent_turns = await get_recent_turns_fn(session_id)
    active_destination = await get_active_destination_fn(session_id)
    active_origin = await get_active_origin_fn(session_id)
    pending_agent_context = await get_pending_agent_context_fn(session_id)
    pending_journey_question = await get_pending_journey_question_fn(session_id)
    recent_turns, pending_agent_context, pending_journey_question = await reset_session_for_destination_change(
        session_id=session_id,
        place=place,
        active_destination=active_destination,
        recent_turns=recent_turns,
        pending_agent_context=pending_agent_context,
        pending_journey_question=pending_journey_question,
    )
    origin, effective_question, awaiting_origin, pending_question = resolve_request_context(
        question=question,
        last_reply=last_reply,
        last_user=last_user,
        pending_agent_context=pending_agent_context,
        pending_journey_question=pending_journey_question,
        active_origin=active_origin,
    )
    same_destination_followup = has_same_destination_followup(
        question=question,
        place=place,
        active_destination=active_destination,
        last_reply=last_reply,
        recent_turns=recent_turns,
        pending_agent_context=pending_agent_context,
        pending_journey_question=pending_journey_question,
    )
    answer_mode = await answer_mode_resolver(
        question=effective_question,
        last_reply=last_reply,
        recent_turns=recent_turns,
        pending_agent_context=pending_agent_context,
        place=place,
    )
    if awaiting_origin and origin and (pending_agent_context or {}).get("mode") == "journey_planning":
        answer_mode = "journey_planning"
    origin = finalize_origin(
        origin=origin, effective_question=effective_question, last_reply=last_reply, active_origin=active_origin
    )
    if origin:
        await set_active_origin_fn(session_id, origin)
    return PreparedRequest(
        question=question,
        effective_question=effective_question,
        origin=origin,
        answer_mode=answer_mode,
        route_or_transport=asks_route_or_transport(effective_question),
        last_user=last_user,
        last_reply=last_reply,
        recent_turns=recent_turns,
        same_destination_followup=same_destination_followup,
        pending_question=pending_question,
    )
