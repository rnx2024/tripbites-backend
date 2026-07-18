# app/location/resolve_country.py
from __future__ import annotations

import httpx

from app.settings import settings


def resolve_country_code(place: str) -> str | None:
    """
    Resolve a city/place name into a 2-letter country code (ISO-3166)
    using Open-Meteo geocoding.
    Returns None if the place cannot be resolved.
    """
    try:
        response = httpx.get(
            settings.openmeteo_geocode_url,
            params={"name": place, "count": 1},
            timeout=5.0,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        return None
    except httpx.HTTPError:
        return None

    try:
        data = response.json()
    except ValueError:
        return None

    results = data.get("results")
    if not results:
        return None
    return results[0].get("country_code", None)
