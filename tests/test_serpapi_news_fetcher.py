import unittest
from unittest.mock import patch

from app.news.serpapi_news_fetcher import fetch_news_items


class SerpApiNewsFetcherTests(unittest.TestCase):
    def test_fetch_news_items_preserves_snippet(self) -> None:
        payload = {
            "news_results": [
                {
                    "title": "Airport shuttle delays after runway works",
                    "snippet": "Mactan terminal transfers are taking longer than usual during overnight repairs.",
                    "source": {"name": "Local News"},
                    "date": "1 hour ago",
                    "link": "https://example.com/airport-delays",
                }
            ]
        }

        with (
            patch("app.news.serpapi_news_fetcher.resolve_country_code", return_value="PH"),
            patch("app.news.serpapi_news_fetcher.get_json_with_retry", return_value=(payload, "")),
        ):
            items, err = fetch_news_items("Cebu")

        self.assertEqual(err, "")
        self.assertEqual(len(items), 1)
        self.assertEqual(
            items[0]["snippet"],
            "Mactan terminal transfers are taking longer than usual during overnight repairs.",
        )


if __name__ == "__main__":
    unittest.main()
