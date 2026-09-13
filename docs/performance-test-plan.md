# Backend Performance Test Plan

## Purpose

Measure local request latency and failure behavior without sending load to the free Render service or consuming external provider quotas.

## Workloads

- `/health/live`: server baseline; safe for repeated local requests.
- `/weather` and `/news`: run locally with controlled provider fakes when measuring application behavior; these routes require `x-api-key`.
- `/chat`: run locally with mocked LLM/provider calls and valid session headers when measuring the application path.
- Cold start: start the local container and record the first request separately from warm requests.

## Example

```bash
uv run python scripts/load_test.py --url http://127.0.0.1:8000 --endpoint health/live --requests 50 --concurrency 5 --output .tmp/performance/health-live.json
```

The report includes total and successful requests, failures, average/p50/p95/maximum latency, requests per second, and grouped error categories. The script is a bounded GET client and does not create sessions or send POST request bodies.

## Safety

Performance reports are written to `.tmp/performance/`, which is ignored by Git. Do not run concurrency tests against production. In particular, do not repeatedly call `/chat`, `/news`, or `/weather` on Render because they can wake or overload the free instance and consume provider quotas.

## Production interpretation

Render captures structured request logs emitted to stdout. Those logs provide real request durations but do not by themselves provide a durable p95 dashboard. Use low-volume `/health/live` probes for availability and export/analyze logs when production latency investigation is needed.
