import unittest
from unittest.mock import Mock, patch

import httpx

from app.news.news_service import search_news
from app.news.tavily_search_fetcher import search_tavily


@patch("time.sleep", return_value=None)
class TavilySearchFetcherTests(unittest.TestCase):
    def test_search_tavily_normalizes_results(self, _sleep) -> None:
        payload = {
            "results": [
                {
                    "title": "IRONMAN 70.3 Davao race weekend schedule",
                    "url": "https://example.com/ironman-davao",
                    "content": "The race is scheduled for Sunday morning with road closures starting earlier.",
                    "published_date": "2026-03-17T03:00:00Z",
                }
            ]
        }
        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None

        with (
            patch("app.news.tavily_search_fetcher.settings.tavily_api", "test-key"),
            patch("app.news.tavily_search_fetcher.httpx.post", return_value=response) as post_mock,
        ):
            items, err = search_tavily("IRONMAN 70.3 Davao schedule", "Davao")

        self.assertEqual(err, "")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "IRONMAN 70.3 Davao race weekend schedule")
        self.assertEqual(items[0]["source"], "example.com")
        self.assertEqual(items[0]["link"], "https://example.com/ironman-davao")
        self.assertIn("Sunday morning", items[0]["snippet"])
        payload_sent = post_mock.call_args.kwargs["json"]
        self.assertIn("IRONMAN 70.3 Davao schedule", payload_sent["query"])
        self.assertEqual(payload_sent["time_range"], "week")

    def test_search_tavily_returns_error_when_api_key_missing(self, _sleep) -> None:
        with patch("app.news.tavily_search_fetcher.settings.tavily_api", ""):
            items, err = search_tavily("Davao race schedule")

        self.assertEqual(items, [])
        self.assertEqual(err, "missing_tavily_api")

    def test_search_news_uses_tavily_provider(self, _sleep) -> None:
        expected = (
            [
                {
                    "title": "Manila cash aid weekend result",
                    "source": "example.com",
                    "date": None,
                    "link": "https://example.com",
                    "snippet": "snippet",
                }
            ],
            "",
        )
        with patch("app.news.news_service.search_tavily", return_value=expected) as search_mock:
            items, err = search_news("cash aid weekend", "Manila")

        self.assertEqual(err, "")
        self.assertEqual(items[0]["title"], "Manila cash aid weekend result")
        search_mock.assert_called_once_with("cash aid weekend", "Manila")

    def test_search_tavily_handles_http_error(self, _sleep) -> None:
        request = httpx.Request("POST", "https://api.tavily.com/search")
        response = httpx.Response(500, request=request)
        error = httpx.HTTPStatusError("boom", request=request, response=response)
        failing = Mock()
        failing.raise_for_status.side_effect = error

        with (
            patch("app.news.tavily_search_fetcher.settings.tavily_api", "test-key"),
            patch("app.news.tavily_search_fetcher.httpx.post", return_value=failing),
        ):
            items, err = search_tavily("Davao event schedule")

        self.assertEqual(items, [])
        self.assertEqual(err, "500")


if __name__ == "__main__":
    unittest.main()
