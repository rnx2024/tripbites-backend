from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.news.news_relevance import (
    HTTPS_SCHEME,
    contains_high_impact_claim,
    contains_https_url,
    meaningful_tokens,
    sanitize_answer_links,
    supports_high_impact_claim,
)


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

    return {"place": place, "final": "", "risk_level": risk_level, "travel_advice": [], "sources": sources}


def _ground_final_answer(final: str, place: str, brief: dict[str, Any]) -> str:
    news_items = brief.get("news_items") or []
    allowed_links = {
        str(item.get("link") or "").strip() for item in news_items if isinstance(item, dict) and item.get("link")
    }
    final = sanitize_answer_links(final, allowed_links)
    if (
        final
        and contains_high_impact_claim(final)
        and not any(isinstance(item, dict) and supports_high_impact_claim(final, item, place) for item in news_items)
    ):
        return f"I couldn't confirm that specific update for {place} from the available news."
    return final


def _news_item_tokens(item: dict[str, Any]) -> set[str]:
    text = " ".join(str(item.get(field) or "") for field in ("title", "snippet"))
    return meaningful_tokens(text)


def _find_best_news_link(final: str, brief: dict[str, Any]) -> str | None:
    answer_tokens = meaningful_tokens(final)
    best_link: str | None = None
    best_score = 0
    for item in brief.get("news_items") or []:
        if not isinstance(item, dict):
            continue
        link = str(item.get("link") or "").strip()
        if not link.startswith(HTTPS_SCHEME):
            continue
        score = len(answer_tokens & _news_item_tokens(item))
        if score > best_score:
            best_link = link
            best_score = score
    return best_link


def _add_news_source_label(final: str, link: str) -> str:
    raw_source = re.compile(rf"(?i)\bsource:\s*{re.escape(link)}")
    if raw_source.search(final):
        return raw_source.sub(f"[Source]({link})", final, count=1)
    if contains_https_url(final):
        return final
    separator = "" if final.endswith((".", "!", "?")) else "."
    return f"{final}{separator} [Source]({link})"


def _append_news_source_link(final: str, brief: dict[str, Any]) -> str:
    if not final or final.lower().startswith("i couldn't confirm that specific update"):
        return final
    best_link = _find_best_news_link(final, brief)
    return _add_news_source_label(final, best_link) if best_link else final
