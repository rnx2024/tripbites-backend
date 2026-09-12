# Architecture and Operations

## Runtime topology

```text
Frontend proxy
      ↓ x-api-key / session headers
Render Docker container
      ↓
FastAPI routes → LangGraph agent → LLM provider
      ├── Redis sessions and short-lived cache
      ├── Weather providers
      ├── News providers
      └── OpenRouteService for journey planning
```

## Deployment

The backend runs as a non-root Docker container on Render. The process listens on Render's `$PORT`, defaulting to `8080` locally. The container healthcheck targets `/health/live`. `/health/ready` verifies Redis availability.

The free Render service may sleep and require a cold start. Request retries belong to the client; the backend keeps provider calls bounded by timeouts and retry policies.

## State and dependencies

Redis stores session and short-lived cache data. The container filesystem is not durable and must not be used for reports, uploads, or persistent application state. External weather, news, routing, and LLM services are accessed through backend-only credentials.

## Observability

Requests emit structured `request.completed` or `request.failed` events containing the method, path, status code, request ID, and duration. Render collects these logs through stdout. Secrets, session tokens, full prompts, provider responses, and sensitive user content must not be logged.

## Verification and recovery

Run the locked test suite and Docker build before deployment. After deployment, check `/health/live`, then `/health/ready` and a normal frontend request. For a provider failure, rely on the stable error response and retry guidance; for Redis failure, restore the dependency before treating the service as ready.
