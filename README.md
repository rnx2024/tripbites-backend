# TripBites Backend

TripBites is a travel intelligence API for destination briefs, local condition awareness, and disruption-aware city updates.

The backend combines:
- weather signals for near-term travel planning
- recent local news for disruptions, closures, safety issues, and major developments
- a LangGraph-powered assistant that turns those inputs into concise travel-facing summaries

## What The API Does

The current backend is designed to help a frontend answer questions like:
- Is this destination generally fine for travel today?
- What weather conditions could affect outdoor plans or transfers?
- Are there recent disruptions, closures, strikes, protests, or safety-related developments?
- What short travel advice should a user see before heading out?

This is not a full itinerary planner. It is a destination intelligence layer that summarizes what matters most right now.

## Main Endpoints

### `POST /chat`

Returns a travel-oriented destination brief for the selected place. The response includes:
- `place`
- `final`
- `risk_level`
- `travel_advice`
- `sources`

Example shape:

```json
{
  "place": "Cebu",
  "final": "Cebu looks generally fine for travel today. Expect warm conditions with some rain-related delays possible. No major recent local disruptions were identified from the current news scan.",
  "risk_level": "low",
  "travel_advice": [
    "Carry light rain protection",
    "Keep an eye on live traffic or terminal updates as plans develop"
  ],
  "sources": [
    {"type": "weather"},
    {"type": "news"}
  ]
}
```

### `GET /weather`

Returns a concise weather summary plus travel relevance guidance. The endpoint is intended as a quick check before outdoor plans, transfers, or day trips.

### `GET /news`

Returns recent local items with travel-focused framing. The intent is to surface disruptions, closures, transport impacts, safety issues, events, and other major developments that may affect travelers.

### `GET /travel-brief`

Optional product-facing endpoint for frontend consumers that want a single structured travel brief without going through chat.

### `GET /health`

Basic service health check.

## Backend Behavior

The backend keeps the original route structure for compatibility, but the business semantics are travel-first:
- `/chat` produces a destination travel brief rather than a generic city summary
- `/weather` emphasizes planning impact for movement and outdoor activities
- `/news` emphasizes local developments relevant to traveler decisions
- `/travel-brief` exposes the same product framing in a simpler read-model

## Architecture

Core pieces:
- FastAPI for the API surface
- LangGraph for the `/chat` assistant
- Open-Meteo and OpenWeather for weather data
- OpenRouteService for routing (set `ORS_API` to enable journey planning routes)
- SerpAPI Google News for recent local reporting
- Redis-backed session and cache helpers
- rate limiting and retry helpers around outbound provider calls

## Security And Access

Protected endpoints require `x-api-key`.

Session-aware chat requests also use the signed TripBites session flow exposed through `POST /session`.

## Local Development

```bash
uv sync
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

### Environment Variables

Set these in a local `.env` file (see `.env.example` for the full template; never commit `.env`):

| Variable | Description |
| --- | --- |
| `OPENROUTER_API_KEY` | LLM provider key used by the LangGraph agent |
| `OPENWEATHER_API_KEY` | OpenWeather current-conditions API key |
| `SERP_API_KEY` | SerpAPI key for Google News fetches |
| `TAVILY_API` | Tavily search API key (news fallback provider) |
| `ORS_API` | OpenRouteService API key; enables journey/route planning |
| `API_KEY` | Shared secret required via `x-api-key` on protected endpoints |
| `SESSION_SECRET` | Signing key for the session token (`itsdangerous`) |
| `FRONTEND_CORS_ORIGIN` | Comma-separated list of allowed CORS origins |
| `REDIS_URL` | Redis connection string for session + cache storage |
| `ENV` | Set to `production` on the production deployment to disable `/docs`, `/redoc`, and `/openapi.json`. Leave unset for local development. |

### Linting, Formatting, and Tests

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest            # run the test suite
```

## Deployment

Deployed as a Docker container on [Render](https://render.com) (see `Dockerfile`; the container binds to `$PORT` if set, falling back to `8080` for local runs).

**Important:** set `ENV=production` in the Render service's environment variables. This is what disables `/docs`, `/redoc`, and `/openapi.json` in production — without it, the API docs stay publicly exposed.
