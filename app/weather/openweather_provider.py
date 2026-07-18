# app/weather/openweather_provider.py
from __future__ import annotations

from typing import Any

from app.http.http_utils import get_json_with_retry
from app.settings import settings


def get_weather_raw(place: str) -> tuple[dict[str, Any], str]:
    """
    Call OpenWeather current weather endpoint.
    URL is loaded from settings.openweather_current_url.
    """
    return get_json_with_retry(
        settings.openweather_current_url,
        {"q": place, "appid": settings.openweather_api_key, "units": "metric"},
    )


def get_weather_line(place: str) -> tuple[str, str]:
    """
    Backwards-compatible one-line weather summary.
    """
    data, err = get_weather_raw(place)
    if err:
        return ("", f"Weather error: {err}")

    name = data.get("name") or place
    wx = (data.get("weather") or [{}])[0]
    desc = wx.get("description") or "n/a"
    temp = (data.get("main") or {}).get("temp", "n/a")

    return (f"{name}: {desc}, {temp}°C", "")
