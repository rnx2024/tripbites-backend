import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app.http.http_client import get_json_with_retry
from app.routing import ors_service
from app.tooling.retry_policy import classifier
from app.weather import openmeteo_provider
from retryguard.integrations.tenacity import wait_retryguard


def _http_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status, request=request, headers=headers or {})
    return httpx.HTTPStatusError("error", request=request, response=response)


class FakeResponse:
    def __init__(self, status: int, headers: dict | None = None, json_data: dict | None = None):
        self.status_code = status
        self.headers = headers or {}
        self._json_data = json_data if json_data is not None else {"ok": True}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _http_error(self.status_code, self.headers)

    def json(self) -> dict:
        return self._json_data


def _fake_get_sequence(*responses: FakeResponse):
    """side_effect callable that yields each FakeResponse in order, repeating the last one after."""
    remaining = list(responses)

    def _get(*args, **kwargs):
        return remaining.pop(0) if remaining else responses[-1]

    return _get


class ErrorClassifierTests(unittest.TestCase):
    """Direct unit tests for retryguard's classification, independent of tenacity/HTTP plumbing."""

    def test_5xx_is_retryable(self) -> None:
        decision = classifier.classify(_http_error(500))
        self.assertTrue(decision.retryable)
        self.assertEqual(decision.category.value, "server")

    def test_404_is_not_retryable(self) -> None:
        decision = classifier.classify(_http_error(404))
        self.assertFalse(decision.retryable)
        self.assertEqual(decision.category.value, "client")

    def test_401_is_not_retryable(self) -> None:
        decision = classifier.classify(_http_error(401))
        self.assertFalse(decision.retryable)
        self.assertEqual(decision.category.value, "auth")

    def test_429_is_retryable(self) -> None:
        decision = classifier.classify(_http_error(429))
        self.assertTrue(decision.retryable)
        self.assertEqual(decision.category.value, "rate_limit")

    def test_429_extracts_retry_after_header(self) -> None:
        decision = classifier.classify(_http_error(429, headers={"Retry-After": "3"}))
        self.assertTrue(decision.retryable)
        self.assertEqual(decision.retry_after_seconds, 3.0)

    def test_httpx_timeout_is_retryable(self) -> None:
        request = httpx.Request("GET", "https://example.com")
        decision = classifier.classify(httpx.ConnectTimeout("timed out", request=request))
        self.assertTrue(decision.retryable)
        self.assertEqual(decision.category.value, "timeout")

    def test_value_error_is_not_retryable(self) -> None:
        decision = classifier.classify(ValueError("bad input"))
        self.assertFalse(decision.retryable)
        self.assertEqual(decision.category.value, "validation")


class WaitRetryguardTests(unittest.TestCase):
    """Direct unit test that the tenacity wait strategy actually reads Retry-After."""

    def test_wait_uses_retry_after_header_over_fallback(self) -> None:
        wait = wait_retryguard(classifier, fallback_seconds=0.5)
        exc = _http_error(429, headers={"Retry-After": "3"})
        fake_state = SimpleNamespace(outcome=SimpleNamespace(failed=True, exception=lambda: exc))

        delay = wait(fake_state)

        self.assertEqual(delay, 3.0)

    def test_wait_falls_back_when_no_retry_after(self) -> None:
        wait = wait_retryguard(classifier, fallback_seconds=0.5)
        exc = _http_error(500)
        fake_state = SimpleNamespace(outcome=SimpleNamespace(failed=True, exception=lambda: exc))

        delay = wait(fake_state)

        self.assertEqual(delay, 2.0)  # classify_http_status suggested_delay_seconds for 5xx


@patch("time.sleep", return_value=None)
class HttpClientRetryTests(unittest.TestCase):
    def test_transient_500_then_200_recovers(self, _sleep) -> None:
        responses = (FakeResponse(500), FakeResponse(500), FakeResponse(200))
        with patch("httpx.get", side_effect=_fake_get_sequence(*responses)) as mock_get:
            data, err = get_json_with_retry("https://example.com/api", {})

        self.assertEqual(data, {"ok": True})
        self.assertEqual(err, "")
        self.assertEqual(mock_get.call_count, 3)

    def test_404_fails_fast_without_retrying(self, _sleep) -> None:
        with patch("httpx.get", side_effect=_fake_get_sequence(FakeResponse(404))) as mock_get:
            data, err = get_json_with_retry("https://example.com/api", {})

        self.assertEqual(data, {})
        self.assertEqual(err, "404")
        self.assertEqual(mock_get.call_count, 1)

    def test_persistent_500_exhausts_default_retries(self, _sleep) -> None:
        responses = (FakeResponse(500), FakeResponse(500), FakeResponse(500))
        with patch("httpx.get", side_effect=_fake_get_sequence(*responses)) as mock_get:
            data, err = get_json_with_retry("https://example.com/api", {}, retries=3)

        self.assertEqual(data, {})
        self.assertEqual(err, "500")
        self.assertEqual(mock_get.call_count, 3)

    def test_retries_param_is_honored(self, _sleep) -> None:
        responses = tuple(FakeResponse(500) for _ in range(5))
        with patch("httpx.get", side_effect=_fake_get_sequence(*responses)) as mock_get:
            get_json_with_retry("https://example.com/api", {}, retries=5)

        self.assertEqual(mock_get.call_count, 5)


@patch("time.sleep", return_value=None)
class OrsServiceRetryTests(unittest.TestCase):
    def test_fetch_route_retries_transient_failure_and_recovers(self, _sleep) -> None:
        good = FakeResponse(
            200,
            json_data={"features": [{"properties": {"summary": {"distance": 1000, "duration": 600}}}]},
        )
        with patch("app.settings.settings.ors_api", "fake-key"):
            with patch("httpx.get", side_effect=_fake_get_sequence(FakeResponse(503), good)) as mock_get:
                route, err = ors_service._fetch_route("driving-car", (0.0, 0.0), (1.0, 1.0))

        self.assertIsNotNone(route)
        self.assertEqual(mock_get.call_count, 2)

    def test_fetch_route_does_not_retry_client_error(self, _sleep) -> None:
        with patch("app.settings.settings.ors_api", "fake-key"):
            with patch("httpx.get", side_effect=_fake_get_sequence(FakeResponse(404))) as mock_get:
                route, err = ors_service._fetch_route("driving-car", (0.0, 0.0), (1.0, 1.0))

        self.assertIsNone(route)
        self.assertEqual(err, "404")
        self.assertEqual(mock_get.call_count, 1)


@patch("time.sleep", return_value=None)
class OpenMeteoProviderRetryTests(unittest.TestCase):
    def test_geocode_place_retries_transient_failure_and_recovers(self, _sleep) -> None:
        good = FakeResponse(
            200,
            json_data={"results": [{"name": "Cebu", "country": "PH", "latitude": 10.3, "longitude": 123.9}]},
        )
        with patch("httpx.get", side_effect=_fake_get_sequence(FakeResponse(500), good)) as mock_get:
            loc, err = openmeteo_provider.geocode_place("Cebu")

        self.assertIsNotNone(loc)
        self.assertEqual(mock_get.call_count, 2)

    def test_geocode_place_does_not_retry_client_error(self, _sleep) -> None:
        with patch("httpx.get", side_effect=_fake_get_sequence(FakeResponse(404))) as mock_get:
            loc, err = openmeteo_provider.geocode_place("Cebu")

        self.assertIsNone(loc)
        self.assertEqual(mock_get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
