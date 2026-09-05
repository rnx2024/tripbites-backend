from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.routes import news_endpoint, travel_brief_endpoint, weather_endpoint


def test_weather_endpoint_dispatches_provider_work_off_event_loop() -> None:
    async def exercise() -> None:
        with patch("app.routes.get_weather_line", return_value=("Sunny", "")) as provider:
            result = await weather_endpoint.__wrapped__(None, "Vigan")  # type: ignore[attr-defined,arg-type]
        provider.assert_called_once_with("Vigan")
        assert result.summary == "Sunny"

    asyncio.run(exercise())


def test_news_endpoint_dispatches_provider_work_off_event_loop() -> None:
    async def exercise() -> None:
        with patch("app.routes.get_news_items", return_value=([], "")) as provider:
            result = await news_endpoint.__wrapped__(None, "Vigan")  # type: ignore[attr-defined,arg-type]
        provider.assert_called_once_with("Vigan")
        assert result.recent_count == 0

    asyncio.run(exercise())


def test_travel_brief_endpoint_dispatches_provider_work_off_event_loop() -> None:
    async def exercise() -> None:
        brief = {
            "place": "Vigan",
            "final": "Fine",
            "risk_level": "low",
            "travel_advice": [],
            "sources": [{"type": "weather"}],
        }
        with patch("app.routes.build_travel_brief", return_value=(brief, "")) as builder:
            result = await travel_brief_endpoint.__wrapped__(None, "Vigan")  # type: ignore[attr-defined,arg-type]
        builder.assert_called_once_with("Vigan")
        assert result.place == "Vigan"

    asyncio.run(exercise())
