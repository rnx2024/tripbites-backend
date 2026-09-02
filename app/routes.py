from __future__ import annotations

import hmac
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from app.agent.agent_service import run_agent
from app.news.news_service import get_news_items
from app.session.errors import SESSION_UNAVAILABLE_MESSAGE, SessionStoreUnavailable
from app.session.session_auth import require_session, sign_session
from app.session.session_store import ensure_session_store_ready
from app.settings import settings
from app.tooling.ratelimit import limiter
from app.tooling.usage_limits import enforce_chat_quota
from app.travel_brief import build_travel_brief
from app.validation import MAX_PLACE_LENGTH, MAX_QUESTION_LENGTH, PlaceValue
from app.weather.weather_service import get_weather_line

router = APIRouter()
RiskLevel = Literal["low", "medium", "high"]
SourceType = Literal["weather", "news"]


def require_api_key(
    x_api_key: Annotated[str, Header(..., alias="x-api-key")],
) -> None:
    if not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _raise_session_unavailable() -> None:
    raise HTTPException(status_code=503, detail=SESSION_UNAVAILABLE_MESSAGE)


async def _ensure_session_store_available() -> None:
    try:
        await ensure_session_store_ready()
    except SessionStoreUnavailable:
        _raise_session_unavailable()


class AgentRequest(BaseModel):
    place: str = Field(min_length=1, max_length=MAX_PLACE_LENGTH)
    question: str | None = Field(default=None, max_length=MAX_QUESTION_LENGTH)

    @field_validator("place", "question", mode="before")
    @classmethod
    def strip_input(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class TravelBriefSourceResponse(BaseModel):
    type: SourceType


class NewsItemResponse(BaseModel):
    title: str
    source: str | None = None
    date: str | None = None
    link: str | None = None
    snippet: str | None = None


class AgentResponse(BaseModel):
    place: str
    final: str
    risk_level: RiskLevel | None = None
    travel_advice: list[str]
    sources: list[TravelBriefSourceResponse]


class TravelBriefResponse(BaseModel):
    place: str
    final: str
    risk_level: RiskLevel
    travel_advice: list[str]
    sources: list[TravelBriefSourceResponse]


class WeatherResponse(BaseModel):
    place: str
    summary: str
    travel_relevance: str
    travel_advice: list[str]


class NewsResponse(BaseModel):
    place: str
    recent_count: int
    items: list[NewsItemResponse]
    travel_relevance: str
    note: str


@router.get("/health")
@limiter.limit("30/minute")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/health/session",
    tags=["session"],
    responses={503: {"description": SESSION_UNAVAILABLE_MESSAGE}},
)
@limiter.limit("30/minute")
async def session_health(request: Request) -> dict[str, str]:
    await _ensure_session_store_available()
    return {"status": "ok"}


@router.post(
    "/session",
    tags=["session"],
    dependencies=[Depends(require_api_key)],
    responses={503: {"description": SESSION_UNAVAILABLE_MESSAGE}},
)
@limiter.limit("30/minute")
async def create_session(request: Request) -> dict[str, str]:
    await _ensure_session_store_available()
    from uuid import uuid4

    session_id = str(uuid4())
    session_token = sign_session(session_id)
    return {"session_id": session_id, "session_token": session_token}


@router.post(
    "/chat",
    tags=["agent"],
    dependencies=[Depends(require_api_key)],
    responses={503: {"description": SESSION_UNAVAILABLE_MESSAGE}},
)
@limiter.limit("15/minute")
async def agent_endpoint(
    request: Request,
    payload: AgentRequest,
    session_id: Annotated[str, Depends(require_session)],
) -> AgentResponse:
    await _ensure_session_store_available()
    await enforce_chat_quota(session_id)
    question = payload.question or ""

    try:
        result = await run_agent(
            session_id=session_id,
            place=payload.place,
            question=question,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="The travel assistant took too long to respond.") from None
    except SessionStoreUnavailable:
        _raise_session_unavailable()

    return AgentResponse(**result)


@router.get(
    "/travel-brief",
    tags=["travel"],
    dependencies=[Depends(require_api_key)],
    responses={502: {"description": "Travel brief generation failed"}},
)
@limiter.limit("15/minute")
async def travel_brief_endpoint(
    request: Request,
    place: Annotated[PlaceValue, Query(..., description="City or destination name")],
) -> TravelBriefResponse:
    brief, err = build_travel_brief(place)
    if err and not brief["sources"]:
        raise HTTPException(status_code=502, detail=err)
    return TravelBriefResponse(**brief)


@router.get(
    "/weather",
    tags=["weather"],
    dependencies=[Depends(require_api_key)],
    responses={502: {"description": "Weather provider request failed"}},
)
@limiter.limit("15/minute")
async def weather_endpoint(
    request: Request,
    place: Annotated[PlaceValue, Query(..., description="City or place name")],
) -> WeatherResponse:
    line, err = get_weather_line(place)
    if err:
        raise HTTPException(status_code=502, detail=err)
    return WeatherResponse(
        place=place,
        summary=line,
        travel_relevance="Use this as a quick weather check before outdoor plans, transfers, or day trips.",
        travel_advice=["Check the latest forecast again before departure if conditions look unstable"],
    )


@router.get(
    "/news",
    tags=["news"],
    dependencies=[Depends(require_api_key)],
    responses={502: {"description": "News retrieval failed"}},
)
@limiter.limit("15/minute")
async def news_endpoint(
    request: Request,
    place: Annotated[PlaceValue, Query(..., description="City or topic for news search")],
) -> NewsResponse:
    headlines, err = get_news_items(place)
    if err:
        raise HTTPException(status_code=502, detail="News retrieval failed")

    return NewsResponse(
        place=place,
        recent_count=len(headlines),
        items=headlines,
        travel_relevance=(
            "Recent items are intended to help spot disruptions, closures, safety issues, transport impacts, "
            "or major local developments that could affect travelers."
        ),
        note="Showing up to 3 recent local items from the last 7 days.",
    )
