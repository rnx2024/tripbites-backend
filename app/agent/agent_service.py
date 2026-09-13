# app/agent_service.py
from __future__ import annotations

from typing import Any

import structlog
from langchain_openai import ChatOpenAI

from app.agent import agent_execution, agent_followup_dispatch, agent_routing
from app.agent.agent_context import PreparedRequest as _PreparedRequest
from app.agent.agent_context import build_user_prompt as _build_user_prompt
from app.agent.agent_context import prepare_request_context
from app.agent.agent_execution import AgentExecutionDependencies, AgentExecutionRequest
from app.agent.agent_followup_dispatch import FollowupDispatchDependencies, FollowupRequest
from app.agent.agent_policy import AnswerMode
from app.agent.agent_response import _append_news_source_link
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
__all__ = ["run_agent", "_append_news_source_link"]

# LangGraph/LangChain is used here for observable tool execution, composable
# weather/news/routing tools, future streaming support, and provider flexibility.

_llm = ChatOpenAI(
    model=settings.openrouter_model,
    temperature=settings.openrouter_temperature,
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    timeout=settings.agent_timeout_seconds,
    max_retries=0,
)


def _get_react_app(include_weather: bool, include_news: bool):
    return agent_execution.get_react_app(_llm, include_weather, include_news)


async def _resolve_answer_mode(
    *,
    question: str | None,
    last_reply: str | None,
    recent_turns: list[dict[str, str]],
    pending_agent_context: dict[str, str] | None,
    place: str,
) -> AnswerMode:
    return await agent_routing.resolve_answer_mode(
        _llm,
        question=question,
        last_reply=last_reply,
        recent_turns=recent_turns,
        pending_agent_context=pending_agent_context,
        place=place,
        timeout_seconds=settings.agent_timeout_seconds,
    )


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
    return agent_routing.build_policy_lines(
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


async def _handle_pre_agent_paths(**kwargs: Any) -> dict[str, Any] | None:
    request = FollowupRequest(**kwargs)
    dependencies = FollowupDispatchDependencies(
        llm=_llm,
        answer_news=_answer_news_followup,
        answer_weather=_answer_weather_followup,
        answer_general=_answer_general_followup,
        answer_journey=_answer_journey_question,
        mark_tools_called=mark_tools_called,
        set_pending_agent_context=set_pending_agent_context,
        set_pending_journey_question=set_pending_journey_question,
        finalize_result=_finalize_result,
    )
    return await agent_followup_dispatch.handle_pre_agent_paths(
        request,
        dependencies,
    )


async def _invoke_agent_graph(
    app: Any,
    user_prompt: str,
    *,
    session_id: str,
    place: str,
) -> tuple[dict[str, Any], float]:
    return await agent_execution.invoke_agent_graph(
        app,
        user_prompt,
        session_id=session_id,
        place=place,
        recursion_limit=settings.agent_recursion_limit,
        timeout_seconds=settings.agent_timeout_seconds,
        log=log,
    )


async def _run_broad_agent(**kwargs: Any) -> dict[str, Any]:
    request = AgentExecutionRequest(**kwargs)
    dependencies = AgentExecutionDependencies(
        build_user_prompt=_build_user_prompt,
        build_policy_lines=_build_policy_lines,
        get_react_app=_get_react_app,
        invoke_agent_graph=_invoke_agent_graph,
        should_include=should_include,
        mark_tools_called=mark_tools_called,
        set_active_destination=set_active_destination,
        log=log,
    )
    return await agent_execution.run_broad_agent(
        request,
        dependencies,
    )


async def _prepare_request_context(*, session_id: str, place: str, question: str | None) -> _PreparedRequest:
    return await prepare_request_context(
        session_id=session_id,
        place=place,
        question=question,
        answer_mode_resolver=_resolve_answer_mode,
        get_last_exchange_fn=get_last_exchange,
        get_recent_turns_fn=get_recent_turns,
        get_active_destination_fn=get_active_destination,
        get_active_origin_fn=get_active_origin,
        get_pending_agent_context_fn=get_pending_agent_context,
        get_pending_journey_question_fn=get_pending_journey_question,
        set_active_origin_fn=set_active_origin,
    )


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
    Resolve session context, route the request, and return a grounded brief.

    The request flow is:
    1. Load recent conversation, destination, origin, and pending context.
    2. Reset destination-scoped state when the destination changes.
    3. Resolve origin and classify the answer mode, including follow-ups.
    4. Handle deterministic clarification/follow-up paths or run the tool-gated
       LangGraph agent for weather, news, routing, or travel-brief data.
    5. Ground the final response, persist the exchange, and return the API shape.

    The implementation does not expose private model reasoning; debug output
    contains execution metadata only.
    """
    log.info("agent.request.received", session_id=session_id, place=place, has_question=bool(question))
    try:
        prepared = await _prepare_request_context(session_id=session_id, place=place, question=question)

        result = await _handle_pre_agent_paths(
            session_id=session_id,
            place=place,
            question=prepared.question,
            last_reply=prepared.last_reply,
            recent_turns=prepared.recent_turns,
            answer_mode=prepared.answer_mode,
            same_destination_followup=prepared.same_destination_followup,
            effective_question=prepared.effective_question,
            pending_question=prepared.pending_question,
            origin=prepared.origin,
            route_or_transport=prepared.route_or_transport,
            debug=debug,
        )
        if result:
            return result

        return await _run_broad_agent(
            session_id=session_id,
            place=place,
            question=prepared.question,
            effective_question=prepared.effective_question,
            origin=prepared.origin,
            answer_mode=prepared.answer_mode,
            route_or_transport=prepared.route_or_transport,
            last_user=prepared.last_user,
            last_reply=prepared.last_reply,
            recent_turns=prepared.recent_turns,
            debug=debug,
        )
    except SessionStoreUnavailable:
        raise
    except Exception:
        log.exception("agent.request.failed", session_id=session_id, place=place)
        raise
