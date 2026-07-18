# app/session_cache.py (UPDATED: facade; public function names unchanged)
from __future__ import annotations

from app.session.session_store import (
    get_active_destination,
    get_active_origin,
    get_last_exchange,
    get_last_sent_timestamps,
    get_or_set,
    get_pending_agent_context,
    get_pending_journey_question,
    get_recent_turns,
    get_session_state,
    mark_sent,
    mark_tools_called,
    prepare_weather_news,
    set_active_destination,
    set_active_origin,
    set_pending_agent_context,
    set_pending_journey_question,
    should_include,
)

# Keep public names stable
__all__ = [
    "get_session_state",
    "should_include",
    "mark_sent",
    "mark_tools_called",
    "get_active_destination",
    "get_active_origin",
    "get_last_exchange",
    "get_pending_agent_context",
    "get_recent_turns",
    "get_pending_journey_question",
    "get_last_sent_timestamps",
    "get_or_set",
    "prepare_weather_news",
    "set_active_destination",
    "set_active_origin",
    "set_pending_agent_context",
    "set_pending_journey_question",
]
