# Architecture and Operations

## Runtime topology

```text
Frontend proxy
      ↓ x-api-key / session headers
FastAPI routes in the Render Docker container
      ├── /session → signed session token
      ├── /chat → context routing → LangGraph tools → grounded response
      ├── /weather, /news, /travel-brief → provider-backed reads
      └── /health/live and /health/ready → liveness/readiness probes
                    ├── Redis sessions and one-hour tool cache
                    ├── Weather providers
                    ├── News providers
                    ├── OpenRouteService
                    └── OpenRouter LLM
```

## Deployment

The backend runs as a non-root Docker container on Render. The process listens on Render's `$PORT`, defaulting to `8080` locally. The container healthcheck targets `/health/live`. `/health/ready` verifies Redis availability.

The free Render service may sleep and require a cold start. Request retries belong to the client; the backend keeps provider calls bounded by timeouts and retry policies.

## State and dependencies

Redis stores conversation/session state and provider/tool cache data. Cache entries
use a one-hour default TTL and signed session tokens use a 24-hour default TTL.
The container filesystem is not durable and must not be used for reports, uploads,
or persistent application state. External weather, news, routing, and LLM services
are accessed through backend-only credentials.

Protected application routes require `x-api-key`; `/chat` additionally requires
the signed `x-session-id` and `x-session-token` headers. Liveness does not require
Redis, while readiness returns `503` until Redis is reachable.

## Observability

Requests emit structured `request.completed` or `request.failed` events containing the method, path, status code, request ID, and duration. Render collects these logs through stdout. Secrets, session tokens, full prompts, provider responses, and sensitive user content must not be logged.

## Verification and recovery

Run `uv run pytest`, `uv run ruff check .`, and the complete Docker image build
before deployment. After deployment, check `/health/live`, then `/health/ready`
and a normal frontend request. For a provider failure, rely on the stable error
response and retry guidance; for Redis failure, restore the dependency before
treating the service as ready.
