# app/agent_service.py
from __future__ import annotations

import json
import time
from typing import Any, cast

import structlog
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.agent.agent_policy import (
    AnswerMode,
    asks_route_or_transport,
    classify_answer_mode,
    decide_tool_includes,
    detect_force_signals,
    extract_origin,
    is_journey_planning_question,
    is_origin_only_reply,
    needs_followup_reference_clarification,
    needs_origin_clarification,
)
from app.agent.agent_prompts import ANSWER_MODE_ROUTER_SYSTEM_PROMPT, LOCAL_INTELLIGENCE_SYSTEM_PROMPT
from app.agent.agent_tools import city_risk_tool, news_search_tool, news_tool, travel_brief_tool, weather_tool
from app.agent.followup_qa import (
    answer_general_followup as _answer_general_followup,
)
from app.agent.followup_qa import (
    answer_journey_question as _answer_journey_question,
)
from app.agent.followup_qa import (
    answer_news_followup as _answer_news_followup,
)
from app.agent.followup_qa import (
    answer_weather_followup as _answer_weather_followup,
)
from app.session.errors import SessionStoreUnavailable

# session memory (Redis-backed)
from app.session.session_cache import (
    get_active_destination,
    get_active_origin,
    get_last_exchange,
    get_pending_agent_context,
    get_pending_journey_question,
    get_recent_turns,
    mark_tools_called,
    set_active_destination,
    set_active_origin,
    set_pending_agent_context,
    set_pending_journey_question,
    should_include,
)
from app.settings import settings

log = structlog.get_logger(__name__)

# -----------------------------------------------------
# LLM + tools
# -----------------------------------------------------
_llm = ChatOpenAI(
    model=settings.openrouter_model,
    temperature=settings.openrouter_temperature,
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
)

# -----------------------------------------------------
# Tool-gated helpers
# -----------------------------------------------------
_REACT_APP_CACHE: dict[tuple[bool, bool], Any] = {}


def _get_react_app(include_weather: bool, include_news: bool):
    key = (include_weather, include_news)
    app = _REACT_APP_CACHE.get(key)
    if app is not None:
        return app

    gated = [travel_brief_tool, city_risk_tool]
    if include_weather:
        gated.append(weather_tool)
    if include_news:
        gated.append(news_tool)
        gated.append(news_search_tool)

    app = create_agent(model=_llm, tools=gated, system_prompt=LOCAL_INTELLIGENCE_SYSTEM_PROMPT)
    _REACT_APP_CACHE[key] = app
    return app


def _build_user_prompt(place: str, question: str | None, origin: str | None = None) -> str:
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


def _format_recent_turns(recent_turns: list[dict[str, str]]) -> list[str]:
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


def _has_same_destination_followup(
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
    if pending_agent_context or pending_journey_question:
        return True
    if recent_turns:
        return True
    return bool((last_reply or "").strip())


async def _resolve_answer_mode(
    *,
    question: str | None,
    last_reply: str | None,
    recent_turns: list[dict[str, str]],
    pending_agent_context: dict[str, str] | None,
    place: str,
) -> AnswerMode:
    fallback = classify_answer_mode(question, last_reply)
    if fallback == "journey_planning":
        return fallback
    if not question or not recent_turns:
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
        response = await _llm.ainvoke(
            [
                {"role": "system", "content": ANSWER_MODE_ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=True, indent=2)},
            ]
        )
        payload = json.loads(str(getattr(response, "content", "") or "").strip())
    except (TypeError, ValueError):
        return fallback

    mode = str(payload.get("mode") or "").strip()
    if mode in {"travel_brief", "news_followup", "weather_followup", "journey_planning"}:
        return cast(AnswerMode, mode)
    return fallback


def _extract_final_message(messages: list[BaseMessage]) -> str:
    final_text = ""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            final_text = str(msg.content)
    return final_text or ""


def _collect_tool_calls(messages: list[BaseMessage]) -> dict[str, dict[str, Any]]:
    pending: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if not (isinstance(msg, AIMessage) and msg.tool_calls):
            continue
        for tc in msg.tool_calls:
            call_id = tc.get("id")
            if not call_id:
                continue
            pending[call_id] = {
                "tool": tc.get("name"),
                "tool_input": tc.get("args"),
                "observation": None,
            }
    return pending


def _attach_tool_observations(messages: list[BaseMessage], pending: dict[str, dict[str, Any]]) -> None:
    if not pending:
        return
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        call_id = getattr(msg, "tool_call_id", None)
        if call_id and call_id in pending:
            pending[call_id]["observation"] = msg.content


def _build_debug(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    pending_tools = _collect_tool_calls(messages)
    _attach_tool_observations(messages, pending_tools)
    return list(pending_tools.values())


def _extract_called_tools(messages: list[BaseMessage]) -> set[str]:
    called: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name")
                if isinstance(name, str) and name:
                    called.add(name)
    return called


def _extract_tool_outputs(messages: list[BaseMessage]) -> dict[str, str]:
    tool_names_by_call_id = _tool_names_by_call_id(messages)
    return _tool_outputs_from_messages(messages, tool_names_by_call_id)


def _tool_names_by_call_id(messages: list[BaseMessage]) -> dict[str, str]:
    tool_names_by_call_id: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                call_id = tc.get("id")
                name = tc.get("name")
                if isinstance(call_id, str) and isinstance(name, str):
                    tool_names_by_call_id[call_id] = name
    return tool_names_by_call_id


def _tool_outputs_from_messages(
    messages: list[BaseMessage],
    tool_names_by_call_id: dict[str, str],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        call_id = getattr(msg, "tool_call_id", None)
        if not isinstance(call_id, str):
            continue
        tool_name = tool_names_by_call_id.get(call_id)
        if tool_name:
            outputs[tool_name] = str(msg.content)
    return outputs


def _extract_structured_brief(messages: list[BaseMessage], place: str) -> dict[str, Any]:
    tool_outputs = _extract_tool_outputs(messages)
    raw_brief = tool_outputs.get("travel_brief_tool")
    if raw_brief:
        try:
            payload = json.loads(raw_brief)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

    risk_output = tool_outputs.get("city_risk_tool", "")
    risk_level = "low"
    if "Risk level: HIGH" in risk_output:
        risk_level = "high"
    elif "Risk level: MEDIUM" in risk_output:
        risk_level = "medium"

    sources: list[dict[str, str]] = []
    if "travel_brief_tool" in tool_outputs or "weather_tool" in tool_outputs:
        sources.append({"type": "weather"})
    if "travel_brief_tool" in tool_outputs or "news_tool" in tool_outputs:
        sources.append({"type": "news"})

    return {
        "place": place,
        "final": "",
        "risk_level": risk_level,
        "travel_advice": [],
        "sources": sources,
    }


def _build_policy_lines(
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
    policy_lines.extend(_format_recent_turns(recent_turns))

    common_lines = [
        "- Mention specific locations only if they are explicitly stated in the retrieved news or weather context.",
        "- If evidence is missing or inconclusive, say it is not specified instead of guessing.",
        "- The selected location from the request is the only destination for this turn. Do NOT treat place names mentioned in the chat as a destination switch.",
    ]

    if answer_mode == "news_followup":
        policy_lines.extend(
            [
                "- Answer the user's specific news question directly in 1-3 sentences. Do NOT produce a generic travel brief.",
                "- You MUST call travel_brief_tool exactly once first to inspect current news_items for the selected location.",
                "- If the current news_items already answer the question, answer directly from those titles/snippets and do NOT call any extra search tool.",
                "- If the current news_items do not answer the question, you MUST call news_search_tool exactly once using a short targeted query composed from the issue/topic and the selected location, such as 'PISTON strike Vigan'.",
                "- If the targeted search still does not confirm the answer, say the retrieved news does not specify it.",
                "- Do NOT include generic travel advice, risk level, or weather unless the user explicitly asked for them.",
            ]
        )
    elif answer_mode == "weather_followup":
        policy_lines.extend(
            [
                "- Answer the user's specific weather question directly in 1-3 sentences. Do NOT produce a generic travel brief.",
                "- You MUST call travel_brief_tool exactly once first to inspect current weather_summary for the selected location.",
                "- If weather_summary already answers the question, answer directly from it and do NOT call weather_tool.",
                "- If weather_summary does not answer the question, you MAY call weather_tool once using the narrowest relevant horizon from the question.",
                "- If the current forecast still does not specify the requested detail, say the current weather data does not specify it.",
                "- Do NOT include generic travel advice, risk level, or unrelated news unless the user explicitly asked for them.",
            ]
        )
    elif answer_mode == "journey_planning":
        policy_lines.extend(
            [
                "- Answer as a journey assessment, not as a destination-only travel brief.",
                "- You MUST call travel_brief_tool exactly once first for the selected destination.",
                f"- Treat '{origin or 'the departure location'}' as the trip origin and '{place}' as the destination.",
                "- If origin is available, inspect origin-side conditions with weather_tool and/or news_tool when they are needed to answer the journey question.",
                "- Distinguish departure conditions, destination conditions, and unknown route conditions.",
                "- If the user asks whether they should continue or postpone the trip, state clearly what is known for the departure point and destination, then note any unknowns along the route.",
                "- Do NOT claim a best route or best transport option from weather/news alone. If asked, say you can comment on likely disruptions and conditions, but not optimize the route without dedicated routing or transport data.",
                "- Keep the answer concise and practical. Do NOT include generic travel-advice bullets or a risk label unless the user explicitly asks for a broad travel brief.",
            ]
        )
        if route_or_transport:
            policy_lines.append(
                "- The user is asking about route or transport choice. Provide only limited guidance from weather/news at the origin and destination, and explicitly say dedicated routing data is not available."
            )
    else:
        policy_lines.extend(
            [
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
        )

    policy_lines.extend(common_lines)
    return policy_lines


async def _reset_session_for_destination_change(
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


def _resolve_origin_context(
    *,
    question: str | None,
    last_reply: str | None,
    last_user: str | None,
    pending_agent_context: dict[str, str] | None,
    pending_journey_question: str | None,
    active_origin: str | None,
) -> tuple[str | None, str | None, bool, str | None]:
    origin = extract_origin(question, last_reply)
    origin = _apply_active_origin(origin, active_origin, question)
    pending_question = (pending_agent_context or {}).get("question")
    awaiting_origin = (pending_agent_context or {}).get("awaiting") == "origin"
    effective_question = question

    if awaiting_origin:
        origin = origin or extract_origin(question, "Where are you traveling from?")
        if origin:
            effective_question = pending_question or pending_journey_question or last_user or question

    if origin and "where are you traveling from" in (last_reply or "").lower() and not effective_question:
        effective_question = pending_journey_question or last_user or question

    if (
        origin
        and effective_question == question
        and is_origin_only_reply(question)
        and last_user
        and (is_journey_planning_question(last_user) or asks_route_or_transport(last_user))
    ):
        effective_question = last_user

    return origin, effective_question, awaiting_origin, pending_question


def _finalize_origin(
    *,
    origin: str | None,
    effective_question: str | None,
    last_reply: str | None,
    active_origin: str | None,
) -> str | None:
    origin = origin or extract_origin(effective_question, last_reply)
    return _apply_active_origin(origin, active_origin, effective_question)


async def _finalize_result(
    *,
    session_id: str,
    place: str,
    question: str | None,
    result: dict[str, Any],
    debug: bool,
) -> dict[str, Any]:
    await mark_tools_called(
        session_id,
        tool_names=[],
        user_message=question,
        agent_reply=result["final"],
    )
    await set_active_destination(session_id, place)
    if debug:
        result["debug"] = []
    return result


async def _maybe_handle_followup_reference(
    *,
    session_id: str,
    place: str,
    question: str | None,
    last_reply: str | None,
    debug: bool,
) -> dict[str, Any] | None:
    if not needs_followup_reference_clarification(question, last_reply):
        return None

    clarification = "I need the specific news item or the previous message to answer that follow-up directly."
    await mark_tools_called(
        session_id,
        tool_names=[],
        user_message=question,
        agent_reply=clarification,
    )
    result: dict[str, Any] = {
        "place": place,
        "final": clarification,
        "risk_level": None,
        "travel_advice": [],
        "sources": [],
    }
    if debug:
        result["debug"] = []
    return result


async def _maybe_handle_followup_modes(
    *,
    session_id: str,
    place: str,
    question: str | None,
    last_reply: str | None,
    recent_turns: list[dict[str, str]],
    answer_mode: AnswerMode,
    same_destination_followup: bool,
    debug: bool,
) -> dict[str, Any] | None:
    if answer_mode == "news_followup":
        result = await _answer_news_followup(
            _llm,
            place,
            question or "",
            last_reply,
            conversation_history=recent_turns,
        )
        return await _finalize_result(
            session_id=session_id,
            place=place,
            question=question,
            result=result,
            debug=debug,
        )

    if answer_mode == "weather_followup":
        result = await _answer_weather_followup(
            _llm,
            place,
            question or "",
            conversation_history=recent_turns,
        )
        return await _finalize_result(
            session_id=session_id,
            place=place,
            question=question,
            result=result,
            debug=debug,
        )

    if same_destination_followup and answer_mode not in {
        "news_followup",
        "weather_followup",
        "journey_planning",
    }:
        result = await _answer_general_followup(
            _llm,
            place,
            question or "",
            last_reply,
            conversation_history=recent_turns,
        )
        return await _finalize_result(
            session_id=session_id,
            place=place,
            question=question,
            result=result,
            debug=debug,
        )

    return None


async def _maybe_handle_journey_mode(
    *,
    session_id: str,
    place: str,
    question: str | None,
    effective_question: str | None,
    last_reply: str | None,
    recent_turns: list[dict[str, str]],
    pending_question: str | None,
    answer_mode: AnswerMode,
    origin: str | None,
    route_or_transport: bool,
    debug: bool,
) -> dict[str, Any] | None:
    if answer_mode == "journey_planning" and needs_origin_clarification(question, last_reply):
        clarification = (
            f"I can assess conditions in {place}, but I need your departure location to judge the trip itself. "
            "Where are you traveling from?"
        )
        await set_pending_agent_context(
            session_id,
            {
                "mode": "journey_planning",
                "awaiting": "origin",
                "question": question or "",
                "destination": place,
            },
        )
        await set_pending_journey_question(session_id, question)
        result: dict[str, Any] = {
            "place": place,
            "final": clarification,
            "risk_level": None,
            "travel_advice": [],
            "sources": [],
        }
        return await _finalize_result(
            session_id=session_id,
            place=place,
            question=question,
            result=result,
            debug=debug,
        )

    if answer_mode == "journey_planning" and origin:
        await set_pending_agent_context(session_id, None)
        await set_pending_journey_question(session_id, None)
        result = await _answer_journey_question(
            _llm,
            place,
            effective_question or question or "",
            origin,
            route_or_transport=route_or_transport,
            latest_user_message=question or "",
            conversation_history=recent_turns,
            pending_question=pending_question,
        )
        return await _finalize_result(
            session_id=session_id,
            place=place,
            question=question,
            result=result,
            debug=debug,
        )

    return None


def _apply_followup_tool_includes(
    answer_mode: AnswerMode,
    include_weather: bool,
    include_news: bool,
) -> tuple[bool, bool]:
    if answer_mode == "news_followup":
        return include_weather, True
    if answer_mode == "weather_followup":
        return True, include_news
    if answer_mode == "journey_planning":
        return True, True
    return include_weather, include_news


async def _handle_pre_agent_paths(
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
) -> dict[str, Any] | None:
    result = await _maybe_handle_followup_reference(
        session_id=session_id,
        place=place,
        question=question,
        last_reply=last_reply,
        debug=debug,
    )
    if result:
        return result

    result = await _maybe_handle_followup_modes(
        session_id=session_id,
        place=place,
        question=question,
        last_reply=last_reply,
        recent_turns=recent_turns,
        answer_mode=answer_mode,
        same_destination_followup=same_destination_followup,
        debug=debug,
    )
    if result:
        return result

    return await _maybe_handle_journey_mode(
        session_id=session_id,
        place=place,
        question=question,
        effective_question=effective_question,
        last_reply=last_reply,
        recent_turns=recent_turns,
        pending_question=pending_question,
        answer_mode=answer_mode,
        origin=origin,
        route_or_transport=route_or_transport,
        debug=debug,
    )


async def _invoke_agent_graph(
    app: Any,
    user_prompt: str,
    *,
    session_id: str,
    place: str,
) -> tuple[dict[str, Any], float]:
    start = time.monotonic()
    try:
        state: dict[str, Any] = await app.ainvoke({"messages": [{"role": "user", "content": user_prompt}]})
    except Exception:
        log.exception("agent.llm_invoke.failed", session_id=session_id, place=place)
        raise
    duration_ms = round((time.monotonic() - start) * 1000, 1)
    return state, duration_ms


async def _run_broad_agent(
    *,
    session_id: str,
    place: str,
    question: str | None,
    effective_question: str | None,
    origin: str | None,
    answer_mode: AnswerMode,
    route_or_transport: bool,
    last_user: str | None,
    last_reply: str | None,
    recent_turns: list[dict[str, str]],
    debug: bool,
) -> dict[str, Any]:
    user_prompt = _build_user_prompt(place, effective_question, origin)

    include_weather, include_news = decide_tool_includes(effective_question)
    force_weather, force_news = detect_force_signals(effective_question or "")
    allow_weather, allow_news = await should_include(session_id, force_weather, force_news)
    include_weather, include_news = _apply_followup_tool_includes(
        answer_mode,
        include_weather,
        include_news,
    )

    if include_weather and not allow_weather and answer_mode == "travel_brief":
        include_weather = False
    if include_news and not allow_news and answer_mode == "travel_brief":
        include_news = False

    policy_lines = _build_policy_lines(
        place=place,
        answer_mode=answer_mode,
        include_weather=include_weather,
        include_news=include_news,
        last_user=last_user,
        last_reply=last_reply,
        recent_turns=recent_turns,
        origin=origin,
        route_or_transport=route_or_transport,
    )

    user_prompt = "\n".join(policy_lines) + "\n\n---\n\n" + user_prompt
    app = _get_react_app(include_weather=include_weather, include_news=include_news)

    state, duration_ms = await _invoke_agent_graph(app, user_prompt, session_id=session_id, place=place)

    messages = state.get("messages", []) or []
    final_text = _extract_final_message(messages)

    called_tools = _extract_called_tools(messages)
    log.info(
        "agent.llm_invoke.completed",
        session_id=session_id,
        place=place,
        duration_ms=duration_ms,
        called_tools=sorted(called_tools),
    )
    await mark_tools_called(
        session_id,
        tool_names=called_tools,
        user_message=question,
        agent_reply=final_text,
    )
    await set_active_destination(session_id, place)

    brief = _extract_structured_brief(messages, place)
    result: dict[str, Any] = {
        "place": str(brief.get("place") or place),
        "final": final_text or str(brief.get("final") or ""),
        "risk_level": (str(brief.get("risk_level") or "low") if answer_mode == "travel_brief" else None),
        "travel_advice": cast(list[str], brief.get("travel_advice") or []) if answer_mode == "travel_brief" else [],
        "sources": cast(list[dict[str, str]], brief.get("sources") or []),
    }
    if debug:
        result["debug"] = _build_debug(messages)
    return result


# -----------------------------------------------------
# Public function: run_agent
# -----------------------------------------------------
async def run_agent(
    *,
    session_id: str,
    place: str,
    question: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Run the LangGraph ReAct agent with tool gating per request.
    """
    log.info("agent.request.received", session_id=session_id, place=place, has_question=bool(question))
    try:
        last_user, last_reply = await get_last_exchange(session_id)
        recent_turns = await get_recent_turns(session_id)
        active_destination = await get_active_destination(session_id)
        active_origin = await get_active_origin(session_id)
        pending_agent_context = await get_pending_agent_context(session_id)
        pending_journey_question = await get_pending_journey_question(session_id)
        recent_turns, pending_agent_context, pending_journey_question = await _reset_session_for_destination_change(
            session_id=session_id,
            place=place,
            active_destination=active_destination,
            recent_turns=recent_turns,
            pending_agent_context=pending_agent_context,
            pending_journey_question=pending_journey_question,
        )
        origin, effective_question, awaiting_origin, pending_question = _resolve_origin_context(
            question=question,
            last_reply=last_reply,
            last_user=last_user,
            pending_agent_context=pending_agent_context,
            pending_journey_question=pending_journey_question,
            active_origin=active_origin,
        )
        same_destination_followup = _has_same_destination_followup(
            question=question,
            place=place,
            active_destination=active_destination,
            last_reply=last_reply,
            recent_turns=recent_turns,
            pending_agent_context=pending_agent_context,
            pending_journey_question=pending_journey_question,
        )

        answer_mode = await _resolve_answer_mode(
            question=effective_question,
            last_reply=last_reply,
            recent_turns=recent_turns,
            pending_agent_context=pending_agent_context,
            place=place,
        )
        if awaiting_origin and origin and (pending_agent_context or {}).get("mode") == "journey_planning":
            answer_mode = "journey_planning"

        origin = _finalize_origin(
            origin=origin,
            effective_question=effective_question,
            last_reply=last_reply,
            active_origin=active_origin,
        )
        if origin:
            await set_active_origin(session_id, origin)
        route_or_transport = asks_route_or_transport(effective_question)

        result = await _handle_pre_agent_paths(
            session_id=session_id,
            place=place,
            question=question,
            last_reply=last_reply,
            recent_turns=recent_turns,
            answer_mode=answer_mode,
            same_destination_followup=same_destination_followup,
            effective_question=effective_question,
            pending_question=pending_question,
            origin=origin,
            route_or_transport=route_or_transport,
            debug=debug,
        )
        if result:
            return result

        return await _run_broad_agent(
            session_id=session_id,
            place=place,
            question=question,
            effective_question=effective_question,
            origin=origin,
            answer_mode=answer_mode,
            route_or_transport=route_or_transport,
            last_user=last_user,
            last_reply=last_reply,
            recent_turns=recent_turns,
            debug=debug,
        )
    except SessionStoreUnavailable:
        raise
    except Exception:
        log.exception("agent.request.failed", session_id=session_id, place=place)
        raise
