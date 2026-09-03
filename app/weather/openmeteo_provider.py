# app/weather/openmeteo_provider.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app.provider_schemas import as_list, as_mapping, bounded_text, finite_coordinate
from app.settings import settings
from app.tooling.retry_policy import build_http_retry

# WMO code → readable description
WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_THUNDERSTORM_CODES = {95, 96, 99}
_HEAVY_RAIN_CODES = {82, 65, 67, 75, 86}
_RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 66, 71, 73, 77, 80, 81, 85}
_SNOW_CODES = {71, 73, 75, 77, 85, 86}
_FOG_CODES = {45, 48}
_CLEAR_OR_CLOUDY_CODES = {0, 1, 2, 3}

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def weather_code_to_text(code: int | None) -> str:
    if code is None:
        return "Unknown conditions"
    return WEATHER_CODE_DESCRIPTIONS.get(code, "Unknown conditions")


def classify_weather_code(code: int | None) -> str:
    if code is None:
        return "unknown"
    if code in _THUNDERSTORM_CODES:
        return "thunderstorm"
    if code in _HEAVY_RAIN_CODES:
        return "heavy_rain"
    if code in _RAIN_CODES:
        return "rain"
    if code in _SNOW_CODES:
        return "snow"
    if code in _FOG_CODES:
        return "fog"
    if code in _CLEAR_OR_CLOUDY_CODES:
        return "clear_or_cloudy"
    return "unknown"


def geocode_place(place: str, language: str = "en"):
    @build_http_retry()
    def _get() -> httpx.Response:
        response = httpx.get(
            settings.openmeteo_geocode_url,
            params={"name": place, "count": 1, "language": language, "format": "json"},
            timeout=5,
        )
        response.raise_for_status()
        return response

    try:
        response = _get()
    except httpx.TimeoutException:
        return None, "Open-Meteo geocoding timeout."
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "http_error"
        return None, f"Open-Meteo geocoding error: {status}"
    except httpx.RequestError as exc:
        return None, str(exc)

    try:
        data = response.json() or {}
    except ValueError:
        return None, "Invalid JSON from Open-Meteo geocoding."

    data = as_mapping(data)
    if data is None:
        return None, "Invalid geocoding response."
    results = as_list(data.get("results"))
    if not results:
        return None, "No geocoding results."

    loc = as_mapping(results[0])
    if loc is None:
        return None, "Invalid geocoding result."
    latitude = finite_coordinate(loc.get("latitude"), minimum=-90, maximum=90)
    longitude = finite_coordinate(loc.get("longitude"), minimum=-180, maximum=180)
    name = bounded_text(loc.get("name"), maximum=200)
    country = bounded_text(loc.get("country"), maximum=100)
    if latitude is None or longitude is None or name is None or country is None:
        return None, "Invalid geocoding result."
    return {
        "name": name,
        "country": country,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": bounded_text(loc.get("timezone"), maximum=100) or "auto",
    }, None


def fetch_openmeteo_forecast(lat: float, lon: float, timezone_name: str = "auto"):
    @build_http_retry()
    def _get() -> httpx.Response:
        response = httpx.get(
            settings.openmeteo_forecast_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "timezone": timezone_name,
                "current": ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "apparent_temperature",
                        "precipitation",
                        "wind_speed_10m",
                        "weather_code",
                        "is_day",
                    ]
                ),
                "daily": ",".join(
                    [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_sum",
                        "uv_index_max",
                        "wind_speed_10m_max",
                    ]
                ),
                "forecast_days": 8,
            },
            timeout=8,
        )
        response.raise_for_status()
        return response

    try:
        response = _get()
    except httpx.TimeoutException:
        return None, "Open-Meteo forecast timeout."
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "http_error"
        return None, f"Open-Meteo forecast error: {status}"
    except httpx.RequestError as exc:
        return None, str(exc)

    try:
        data = as_mapping(response.json())
    except ValueError:
        return None, "Invalid JSON from Open-Meteo forecast."
    if data is None or not isinstance(data.get("current"), dict) or not isinstance(data.get("daily"), dict):
        return None, "Invalid Open-Meteo forecast response."
    return data, None


def _pick_daily_value(daily: dict[str, Any], key: str, idx: int):
    arr = daily.get(key) or []
    if isinstance(arr, list) and 0 <= idx < len(arr):
        return arr[idx]
    return None


def _resolve_daily_index(times: Any, horizon: str, target_date: str) -> int:
    if isinstance(times, list) and times:
        try:
            return times.index(target_date)
        except ValueError:
            idx = 0 if horizon in ("now", "today") else 1
            return max(0, min(idx, len(times) - 1))

    return 0 if horizon in ("now", "today") else 1


def _to_local_today(timezone_name: str) -> datetime:
    try:
        tz = ZoneInfo(timezone_name) if timezone_name and timezone_name != "auto" else UTC
    except ZoneInfoNotFoundError:
        tz = UTC
    return datetime.now(tz=tz)


def resolve_horizon_to_date_str(horizon: str, timezone_name: str) -> str:
    h = (horizon or "").strip().lower()
    now_local = _to_local_today(timezone_name)
    today = now_local.date()

    if h in ("now", "today", ""):
        return today.isoformat()
    if h == "tomorrow":
        return (today + timedelta(days=1)).isoformat()

    if h in _WEEKDAYS:
        target = _WEEKDAYS[h]
        delta = (target - today.weekday()) % 7
        return (today + timedelta(days=delta)).isoformat()

    # absolute ISO date passthrough (YYYY-MM-DD)
    if len(h) == 10 and h[4] == "-" and h[7] == "-":
        return h

    return today.isoformat()


def get_weather_summary(place: str, horizon: str = "today", language: str = "en"):
    loc, err = geocode_place(place, language)
    if err or not loc:
        return None, err

    lat = loc.get("latitude")
    lon = loc.get("longitude")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None, "Invalid geocoding coordinates."

    normalized_horizon = (horizon or "").strip().lower()
    raw, werr = fetch_openmeteo_forecast(
        lat=float(lat),
        lon=float(lon),
        timezone_name=loc.get("timezone") or "auto",
    )
    if werr or not raw:
        return None, werr

    current = raw.get("current") or {}
    daily = raw.get("daily") or {}

    times = daily.get("time") or []
    target_date = resolve_horizon_to_date_str(normalized_horizon, loc.get("timezone") or "auto")
    idx = _resolve_daily_index(times, normalized_horizon, target_date)

    return (
        _build_summary(
            current=current,
            daily=daily,
            target_date=target_date,
            label=f"{loc['name']}, {loc['country']}",
            idx=idx,
        ),
        None,
    )


def _build_summary(
    *,
    current: dict[str, Any],
    daily: dict[str, Any],
    target_date: str,
    label: str,
    idx: int,
) -> dict[str, Any]:
    return {
        "place_label": label,
        "current": {
            "temp_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precip_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "weather_text": weather_code_to_text(current.get("weather_code")),
        },
        "day": {
            "label": target_date,
            "tmin_c": _pick_daily_value(daily, "temperature_2m_min", idx),
            "tmax_c": _pick_daily_value(daily, "temperature_2m_max", idx),
            "precip_mm": _pick_daily_value(daily, "precipitation_sum", idx),
            "uv_index_max": _pick_daily_value(daily, "uv_index_max", idx),
            "wind_speed_max_kmh": _pick_daily_value(daily, "wind_speed_10m_max", idx),
        },
    }


def get_weather_summary_by_coords(
    lat: float,
    lon: float,
    horizon: str = "today",
    *,
    label: str | None = None,
    timezone_name: str = "auto",
):
    normalized_horizon = (horizon or "").strip().lower()
    raw, werr = fetch_openmeteo_forecast(lat=float(lat), lon=float(lon), timezone_name=timezone_name)
    if werr or not raw:
        return None, werr

    current = raw.get("current") or {}
    daily = raw.get("daily") or {}
    target_date = resolve_horizon_to_date_str(normalized_horizon, timezone_name)
    times = daily.get("time") or []
    idx = _resolve_daily_index(times, normalized_horizon, target_date)
    place_label = label or "En route"

    return _build_summary(
        current=current,
        daily=daily,
        target_date=target_date,
        label=place_label,
        idx=idx,
    ), None
