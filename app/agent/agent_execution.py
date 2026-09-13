from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AgentExecutionRequest:
    session_id: str
    place: str
    question: str | None
    effective_question: str | None
    origin: str | None
    answer_mode: AnswerMode
    route_or_transport: bool
    last_user: str | None
    last_reply: str | None
    recent_turns: list[dict[str, str]]
    debug: bool


@dataclass(frozen=True)
class AgentExecutionDependencies:
    build_user_prompt: Callable[..., str]
    build_policy_lines: Callable[..., list[str]]
    get_react_app: Callable[..., Any]
    invoke_agent_graph: Callable[..., Any]
    should_include: Callable[..., Any]
    mark_tools_called: Callable[..., Any]
    set_active_destination: Callable[..., Any]
    log: Any


async def run_broad_agent(
    request: AgentExecutionRequest, dependencies: AgentExecutionDependencies
) -> dict[str, Any]:
    include_weather, include_news = await _resolve_tool_includes(request, dependencies)
    user_prompt = _build_agent_prompt(request, dependencies, include_weather, include_news)
    app = dependencies.get_react_app(include_weather=include_weather, include_news=include_news)
    state, duration_ms = await dependencies.invoke_agent_graph(
        app,
        user_prompt,
        session_id=request.session_id,
        place=request.place,
    )
    return await _build_agent_result(request, dependencies, state, duration_ms)


async def _resolve_tool_includes(
    request: AgentExecutionRequest, dependencies: AgentExecutionDependencies
) -> tuple[bool, bool]:
    include_weather, include_news = decide_tool_includes(request.effective_question)
    force_weather, force_news = detect_force_signals(request.effective_question or "")
    allow_weather, allow_news = await dependencies.should_include(
        request.session_id, force_weather, force_news
    )
    include_weather, include_news = _include_mode_tools(
        request.answer_mode, include_weather, include_news
    )
    if request.answer_mode == "travel_brief":
        include_weather = include_weather and allow_weather
        include_news = include_news and allow_news
    return include_weather, include_news


def _include_mode_tools(
    answer_mode: AnswerMode, include_weather: bool, include_news: bool
) -> tuple[bool, bool]:
    if answer_mode == "news_followup":
        return include_weather, True
    if answer_mode == "weather_followup":
        return True, include_news
    if answer_mode == "journey_planning":
        return True, True
    return include_weather, include_news


def _build_agent_prompt(
    request: AgentExecutionRequest,
    dependencies: AgentExecutionDependencies,
    include_weather: bool,
    include_news: bool,
) -> str:
    user_prompt = dependencies.build_user_prompt(request.place, request.effective_question, request.origin)
    policy_lines = dependencies.build_policy_lines(
        place=request.place,
        answer_mode=request.answer_mode,
        include_weather=include_weather,
        include_news=include_news,
        last_user=request.last_user,
        last_reply=request.last_reply,
        recent_turns=request.recent_turns,
        origin=request.origin,
        route_or_transport=request.route_or_transport,
    )
    return "\n".join(policy_lines) + "\n\n---\n\n" + user_prompt


async def _build_agent_result(
    request: AgentExecutionRequest,
    dependencies: AgentExecutionDependencies,
    state: dict[str, Any],
    duration_ms: float,
) -> dict[str, Any]:
    messages = state.get("messages", []) or []
    final_text = _extract_final_message(messages)
    called_tools = _extract_called_tools(messages)
    dependencies.log.info(
        "agent.llm_invoke.completed",
        session_id=request.session_id,
        place=request.place,
        duration_ms=duration_ms,
        called_tools=sorted(called_tools),
    )
    await dependencies.mark_tools_called(
        request.session_id,
        tool_names=called_tools,
        user_message=request.question,
        agent_reply=final_text,
    )
    await dependencies.set_active_destination(request.session_id, request.place)
    return _shape_agent_result(request, messages, final_text)


def _shape_agent_result(
    request: AgentExecutionRequest, messages: list[Any], final_text: str
) -> dict[str, Any]:
    brief = _extract_structured_brief(messages, request.place)
    grounded_answer = _ground_final_answer(final_text or str(brief.get("final") or ""), request.place, brief)
    result: dict[str, Any] = {
        "place": str(brief.get("place") or request.place),
        "final": _append_news_source_link(grounded_answer, brief),
        "risk_level": str(brief.get("risk_level") or "low") if request.answer_mode == "travel_brief" else None,
        "travel_advice": (
            cast(list[str], brief.get("travel_advice") or [])
            if request.answer_mode == "travel_brief"
            else []
        ),
        "sources": cast(list[dict[str, str]], brief.get("sources") or []),
    }
    if request.debug:
        result["debug"] = _build_debug(messages)
    return result
