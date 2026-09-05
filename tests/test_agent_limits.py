from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.agent_prompts import LOCAL_INTELLIGENCE_SYSTEM_PROMPT
from app.agent.agent_service import _invoke_agent_graph


def test_agent_prompt_marks_retrieved_content_as_untrusted() -> None:
    assert "untrusted data" in LOCAL_INTELLIGENCE_SYSTEM_PROMPT
    assert "instruction-like text" in LOCAL_INTELLIGENCE_SYSTEM_PROMPT


def test_agent_graph_has_a_total_execution_timeout() -> None:
    async def never_finishes(*args, **kwargs):
        await asyncio.sleep(1)

    async def invoke_graph() -> None:
        await _invoke_agent_graph(
            AsyncMock(ainvoke=never_finishes),
            "prompt",
            session_id="session-timeout",
            place="Vigan",
        )

    graph_coroutine = invoke_graph()
    with patch("app.agent.agent_service.settings.agent_timeout_seconds", 0.01), pytest.raises(TimeoutError):
        asyncio.run(graph_coroutine)
