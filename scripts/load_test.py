from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class RequestResult:
    latency_ms: float
    status_code: int | None
    error_category: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded local HTTP performance test.")
    parser.add_argument("--url", required=True, help="Base URL, for example http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="health/live", help="GET endpoint path")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.timeout <= 0:
        parser.error("requests, concurrency, and timeout must be positive")
    args.concurrency = min(args.concurrency, args.requests)
    return args


def _request(client: httpx.Client, url: str) -> RequestResult:
    started = time.perf_counter()
    try:
        response = client.get(url)
        category = None if 200 <= response.status_code < 300 else _status_category(response.status_code)
        return RequestResult(_elapsed_ms(started), response.status_code, category)
    except httpx.TimeoutException:
        return RequestResult(_elapsed_ms(started), None, "timeout")
    except httpx.ConnectError:
        return RequestResult(_elapsed_ms(started), None, "connection_error")
    except httpx.RequestError:
        return RequestResult(_elapsed_ms(started), None, "request_error")
    except Exception:
        return RequestResult(_elapsed_ms(started), None, "unexpected_error")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _status_category(status_code: int) -> str:
    return f"http_{status_code // 100}xx"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def _summarize(results: list[RequestResult], duration_seconds: float, args: argparse.Namespace) -> dict[str, Any]:
    latencies = [result.latency_ms for result in results]
    errors: dict[str, int] = {}
    for result in results:
        if result.error_category:
            errors[result.error_category] = errors.get(result.error_category, 0) + 1
    successful = sum(result.error_category is None for result in results)
    return {
        "target": f"{args.url.rstrip('/')}/{args.endpoint.lstrip('/')}",
        "total_requests": len(results),
        "successful_requests": successful,
        "failed_requests": len(results) - successful,
        "average_latency_ms": round(statistics.fmean(latencies), 2),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "max_latency_ms": max(latencies),
        "requests_per_second": round(len(results) / duration_seconds, 2) if duration_seconds else 0.0,
        "error_categories": errors,
        "duration_seconds": round(duration_seconds, 2),
    }


def main() -> int:
    args = _parse_args()
    url = f"{args.url.rstrip('/')}/{args.endpoint.lstrip('/')}"
    headers = {"x-api-key": args.api_key} if args.api_key else None
    started = time.perf_counter()
    with (
        httpx.Client(headers=headers, timeout=args.timeout) as client,
        ThreadPoolExecutor(max_workers=args.concurrency) as executor,
    ):
        futures = [executor.submit(_request, client, url) for _ in range(args.requests)]
        results = [future.result() for future in as_completed(futures)]
    report = _summarize(results, time.perf_counter() - started, args)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["failed_requests"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
