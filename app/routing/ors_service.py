from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import httpx

from app.settings import settings
from app.tooling.retry_policy import build_http_retry
from app.weather.openmeteo_provider import geocode_place

log = logging.getLogger(__name__)

DEFAULT_PROFILES = (
    "driving-car",
    "cycling-regular",
    "foot-walking",
)

PROFILE_LABELS = {
    "driving-car": "car",
    "cycling-regular": "bike",
    "foot-walking": "walk",
}


def _build_point_label(loc: Dict[str, Any], fallback: str) -> str:
    name = str(loc.get("name") or "").strip()
    country = str(loc.get("country") or "").strip()
    if name and country:
        return f"{name}, {country}"
    if name:
        return name
    return fallback


def _midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    return (lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0


def _fetch_route(profile: str, start: Tuple[float, float], end: Tuple[float, float]) -> Tuple[Dict[str, Any] | None, str]:
    if not settings.ors_api:
        return None, "missing_ors_api"

    url = f"{settings.ors_directions_url.rstrip('/')}/{profile}"
    params = {
        "start": f"{start[1]},{start[0]}",
        "end": f"{end[1]},{end[0]}",
        "format": "geojson",
    }
    headers = {"Authorization": settings.ors_api}

    @build_http_retry()
    def _get() -> httpx.Response:
        response = httpx.get(url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        return response

    try:
        response = _get()
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "http_error"
        return None, str(status)
    except httpx.RequestError as exc:
        return None, str(exc)

    try:
        data = response.json()
    except ValueError:
        return None, "invalid_json"

    features = data.get("features") or []
    if not features:
        return None, "no_route"
    summary = (features[0].get("properties") or {}).get("summary") or {}
    distance_m = summary.get("distance")
    duration_s = summary.get("duration")
    if not isinstance(distance_m, (int, float)) or not isinstance(duration_s, (int, float)):
        return None, "invalid_summary"

    return {
        "profile": profile,
        "mode": PROFILE_LABELS.get(profile, profile),
        "distance_km": round(distance_m / 1000.0, 2),
        "duration_min": round(duration_s / 60.0, 1),
        "raw_distance_m": distance_m,
        "raw_duration_s": duration_s,
    }, ""


def _resolve_location(place: str, role: str) -> Tuple[Dict[str, Any] | None, str]:
    loc, err = geocode_place(place)
    if err or not loc:
        return None, f"{role}_geocode_failed:{err or 'unknown'}"

    lat = loc.get("latitude")
    lon = loc.get("longitude")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None, "invalid_geocoding_coordinates"

    return {
        "loc": loc,
        "lat": float(lat),
        "lon": float(lon),
    }, ""


def _collect_routes(
    profiles: Tuple[str, ...],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    routes: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    for profile in profiles:
        route, err = _fetch_route(profile, start, end)
        if route:
            routes.append(route)
        else:
            errors[profile] = err or "route_error"
    return routes, errors


def _select_best_route(routes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return min(routes, key=lambda r: r.get("raw_duration_s") or float("inf"))


def plan_route(
    origin: str,
    destination: str,
    profiles: Tuple[str, ...] | None = None,
) -> Tuple[Dict[str, Any] | None, str]:
    if not settings.ors_api:
        return None, "missing_ors_api"

    origin_data, oerr = _resolve_location(origin, "origin")
    if oerr or not origin_data:
        return None, oerr or "origin_geocode_failed:unknown"

    dest_data, derr = _resolve_location(destination, "destination")
    if derr or not dest_data:
        return None, derr or "destination_geocode_failed:unknown"

    start = (origin_data["lat"], origin_data["lon"])
    end = (dest_data["lat"], dest_data["lon"])
    midpoint_lat, midpoint_lon = _midpoint(start[0], start[1], end[0], end[1])

    chosen_profiles = profiles or DEFAULT_PROFILES
    routes, errors = _collect_routes(chosen_profiles, start, end)

    if not routes:
        return None, "no_routes"

    best = _select_best_route(routes)
    return {
        "origin": {
            "label": _build_point_label(origin_data["loc"], origin),
            "lat": start[0],
            "lon": start[1],
        },
        "destination": {
            "label": _build_point_label(dest_data["loc"], destination),
            "lat": end[0],
            "lon": end[1],
        },
        "routes": routes,
        "best_profile": best.get("profile"),
        "best_mode": best.get("mode"),
        "best_distance_km": best.get("distance_km"),
        "best_duration_min": best.get("duration_min"),
        "midpoint": {"lat": midpoint_lat, "lon": midpoint_lon},
        "errors": errors,
        "source": "openrouteservice",
    }, ""
