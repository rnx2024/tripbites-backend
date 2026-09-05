from unittest.mock import patch

import httpx

from app.http.http_client import get_json_with_retry


def test_http_client_hides_request_exception_details() -> None:
    request = httpx.Request("GET", "https://provider.example/data")
    failure = httpx.ConnectError("internal host and credential details", request=request)

    with patch("app.http.http_client.httpx.get", side_effect=failure):
        payload, error = get_json_with_retry(
            "https://provider.example/data", params={}, retries=1
        )

    assert payload == {}
    assert error == "request_failed"
    assert "internal host" not in error
