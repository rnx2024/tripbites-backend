from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent

from app.agent.agent_policy import AnswerMode, decide_tool_includes, detect_force_signals
from app.agent.agent_prompts import LOCAL_INTELLIGENCE_SYSTEM_PROMPT
from app.agent.agent_response import (
    _append_news_source_link,
    _build_debug,
    _extract_called_tools,
    _extract_final_message,
    _extract_structured_brief,
    _ground_final_answer,
)
from app.agent.agent_tools import city_risk_tool, news_search_tool, news_tool, travel_brief_tool, weather_tool

_REACT_APP_CACHE: dict[tuple[bool, bool], Any] = {}


def get_react_app(llm: Any, include_weather: bool, include_news: bool) -> Any:
    key = (include_weather, include_news)
    app = _REACT_APP_CACHE.get(key)
    if app is not None:
        return app
    gated = [travel_brief_tool, city_risk_tool]
    if include_weather:
        gated.append(weather_tool)
    if include_news:
        gated.extend([news_tool, news_search_tool])
    app = create_agent(model=llm, tools=gated, system_prompt=LOCAL_INTELLIGENCE_SYSTEM_PROMPT)
    _REACT_APP_CACHE[key] = app
    return app


async def invoke_agent_graph(
    app: Any,
    user_prompt: str,
    *,
    session_id: str,
    place: str,
    recursion_limit: int,
    timeout_seconds: float,
    log: Any,
) -> tuple[dict[str, Any], float]:
    start = time.monotonic()
    try:
        state: dict[str, Any] = await asyncio.wait_for(
            app.ainvoke(
                {"messages": [{"role": "user", "content": user_prompt}]},
                config={"recursion_limit": recursion_limit},
            ),
            timeout=timeout_seconds,
        )
    except Exception:
        log.exception("agent.llm_invoke.failed", session_id=session_id, place=place)
        raise
    return state, round((time.monotonic() - start) * 1000, 1)


async def run_broad_agent(
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
    build_user_prompt_fn: Callable[..., str],
    build_policy_lines_fn: Callable[..., list[str]],
    get_react_app_fn: Callable[..., Any],
    invoke_agent_graph_fn: Callable[..., Any],
    should_include_fn: Callable[..., Any],
    mark_tools_called_fn: Callable[..., Any],
    set_active_destination_fn: Callable[..., Any],
    recursion_limit: int,
    timeout_seconds: float,
    log: Any,
) -> dict[str, Any]:
    user_prompt = build_user_prompt_fn(place, effective_question, origin)
    include_weather, include_news = decide_tool_includes(effective_question)
    force_weather, force_news = detect_force_signals(effective_question or "")
    allow_weather, allow_news = await should_include_fn(session_id, force_weather, force_news)
    if answer_mode == "news_followup":
        include_news = True
    elif answer_mode == "weather_followup":
        include_weather = True
    elif answer_mode == "journey_planning":
        include_weather = include_news = True
    if include_weather and not allow_weather and answer_mode == "travel_brief":
        include_weather = False
    if include_news and not allow_news and answer_mode == "travel_brief":
        include_news = False
    policy_lines = build_policy_lines_fn(
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
    app = get_react_app_fn(include_weather=include_weather, include_news=include_news)
    state, duration_ms = await invoke_agent_graph_fn(
        app,
        "\n".join(policy_lines) + "\n\n---\n\n" + user_prompt,
        session_id=session_id,
        place=place,
    )
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
    await mark_tools_called_fn(session_id, tool_names=called_tools, user_message=question, agent_reply=final_text)
    await set_active_destination_fn(session_id, place)
    brief = _extract_structured_brief(messages, place)
    grounded_answer = _ground_final_answer(final_text or str(brief.get("final") or ""), place, brief)
    grounded_final = _append_news_source_link(grounded_answer, brief)
    return {
        "place": str(brief.get("place") or place),
        "final": grounded_final,
        "risk_level": str(brief.get("risk_level") or "low") if answer_mode == "travel_brief" else None,
        "travel_advice": cast(list[str], brief.get("travel_advice") or []) if answer_mode == "travel_brief" else [],
        "sources": cast(list[dict[str, str]], brief.get("sources") or []),
        **({"debug": _build_debug(messages)} if debug else {}),
    }
