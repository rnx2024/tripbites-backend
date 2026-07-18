from __future__ import annotations

from typing import Any

from app.weather.weather_service import classify_weather_code


def score_weather_risk(summary: Any) -> tuple[int, list[str]]:
    if not summary:
        return 0, []

    current = summary.get("current") or {}
    day = summary.get("day") or {}

    score = 0
    reasons: list[str] = []

    score, reasons = _apply_risk_signal(score, reasons, _score_weather_category(current.get("weather_code")))
    score, reasons = _apply_risk_signal(score, reasons, _score_wind(day.get("wind_speed_max_kmh")))
    score, reasons = _apply_risk_signal(score, reasons, _score_precip(day.get("precip_mm")))
    score, reasons = _apply_risk_signal(score, reasons, _score_temperature(day.get("tmin_c"), day.get("tmax_c")))

    return score, reasons


def _apply_risk_signal(score: int, reasons: list[str], signal: tuple[int, list[str]]) -> tuple[int, list[str]]:
    delta, notes = signal
    if delta:
        score += delta
    if notes:
        reasons.extend(notes)
    return score, reasons


def _score_weather_category(code: Any) -> tuple[int, list[str]]:
    category = classify_weather_code(code)
    mapping = {
        "thunderstorm": (3, ["thunderstorms are expected"]),
        "heavy_rain": (2, ["heavy rain is likely"]),
        "rain": (1, ["scattered rain is possible"]),
        "snow": (2, ["snow or icy conditions may affect movement"]),
        "fog": (1, ["reduced visibility is possible"]),
    }
    return mapping.get(category, (0, []))


def _score_wind(wind_max: Any) -> tuple[int, list[str]]:
    wind = wind_max or 0
    if wind >= 70:
        return 3, ["very strong winds may disrupt transport"]
    if wind >= 50:
        return 2, ["strong winds may affect outdoor plans"]
    if wind >= 30:
        return 1, ["gusty conditions are expected"]
    return 0, []


def _score_precip(precip: Any) -> tuple[int, list[str]]:
    precip_mm = precip or 0
    if precip_mm >= 30:
        return 2, ["heavy rainfall could slow local travel"]
    if precip_mm >= 5:
        return 1, ["rain may affect outdoor timing"]
    return 0, []


def _score_temperature(tmin: Any, tmax: Any) -> tuple[int, list[str]]:
    notes: list[str] = []
    score = 0
    if tmax is not None and tmax >= 35:
        score += 2
        notes.append("very hot conditions are expected")
    if tmin is not None and tmin <= -5:
        score += 2
        notes.append("very cold conditions are expected")
    return score, notes


def classify_risk_level(score: int, *, uppercase: bool = False) -> str:
    if score >= 5:
        return "HIGH" if uppercase else "high"
    if score >= 2:
        return "MEDIUM" if uppercase else "medium"
    return "LOW" if uppercase else "low"
