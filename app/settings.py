# settings.py
from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API keys
    api_key: str
    openrouter_api_key: str
    openweather_api_key: str
    serp_api_key: str
    tavily_api: str
    ors_api: str

    # LLM config
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "gpt-4o-mini"
    openrouter_temperature: float = 0.0

    # External API base URLs
    openweather_current_url: str = "https://api.openweathermap.org/data/2.5/weather"
    openmeteo_geocode_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    openmeteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    serpapi_search_url: str = "https://serpapi.com/search.json"
    tavily_search_url: str = "https://api.tavily.com/search"
    ors_directions_url: str = "https://api.openrouteservice.org/v2/directions"
    frontend_cors_origin: str

    # Database / cache
    redis_url: str
    session_secret: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("*", mode="before")
    @classmethod
    def _strip_required_strings(cls, value):
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("Empty environment values are not allowed")
            return cleaned
        return value

    @field_validator("frontend_cors_origin")
    @classmethod
    def _validate_cors_origins(cls, value: str) -> str:
        from urllib.parse import urlparse

        origins = [item.strip().rstrip("/") for item in value.split(",") if item.strip()]
        if not origins:
            raise ValueError("FRONTEND_CORS_ORIGIN must contain at least one origin")
        for origin in origins:
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path:
                raise ValueError("FRONTEND_CORS_ORIGIN must contain valid origins")
        return ",".join(origins)


settings = Settings()
