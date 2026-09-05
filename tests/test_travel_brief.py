import unittest
from unittest.mock import patch

from app.travel_brief import build_travel_brief


class TravelBriefTests(unittest.TestCase):
    def test_brief_is_grounded_in_weather_and_news_details(self) -> None:
        weather_summary = {
            "place_label": "Cebu, Philippines",
            "current": {
                "weather_code": 63,
                "weather_text": "Moderate rain",
            },
            "day": {
                "label": "2026-03-16",
                "tmin_c": 24,
                "tmax_c": 30,
                "precip_mm": 12,
                "uv_index_max": 6,
                "wind_speed_max_kmh": 32,
            },
        }
        headlines = [
            {
                "title": "Cebu airport shuttle delays after runway works",
                "snippet": "Cebu Mactan terminal transfers are taking longer than usual during overnight repairs.",
                "source": "Local News",
                "date": "2026-03-16T05:00:00+00:00",
                "link": "https://example.com/airport-delays",
            }
        ]

        with (
            patch("app.travel_brief.get_weather_summary", return_value=(weather_summary, None)),
            patch("app.travel_brief.get_news_items", return_value=(headlines, "")),
            patch(
                "app.travel_brief.assess_travel_concern",
                return_value={
                    "risk_level": "high",
                    "final": (
                        "Cebu may be challenging for travel today. Expect moderate rain with temperatures around 24-30°C. "
                        "Airport shuttle delays after runway works may affect transfers."
                    ),
                    "travel_advice": [
                        "Allow extra transit time because airport shuttle delays are affecting transfers."
                    ],
                    "weather_reasons": ["moderate rain may slow local movement"],
                    "news_reasons": ["airport shuttle delays may affect transfers"],
                    "relevant_news_items": headlines,
                },
            ),
        ):
            brief, err = build_travel_brief("Cebu")

        self.assertEqual(err, "")
        self.assertEqual(brief["risk_level"], "high")
        self.assertIn("moderate rain", brief["final"].lower())
        self.assertIn("24-30", brief["final"])
        self.assertIn("Airport shuttle delays after runway works", brief["final"])
        self.assertEqual(brief["news_items"][0]["snippet"], headlines[0]["snippet"])
        self.assertTrue(
            any("airport shuttle delays" in item.lower() for item in brief["travel_advice"]),
            brief["travel_advice"],
        )
        self.assertTrue(brief["weather_reasons"])
        self.assertTrue(brief["news_reasons"])

    def test_brief_does_not_claim_quiet_news_when_news_fetch_failed(self) -> None:
        weather_summary = {
            "place_label": "Cebu, Philippines",
            "current": {
                "weather_code": 1,
                "weather_text": "Mainly clear",
            },
            "day": {
                "label": "2026-03-16",
                "tmin_c": 25,
                "tmax_c": 31,
                "precip_mm": 0,
                "uv_index_max": 5,
                "wind_speed_max_kmh": 12,
            },
        }

        with (
            patch("app.travel_brief.get_weather_summary", return_value=(weather_summary, None)),
            patch("app.travel_brief.get_news_items", return_value=([], "provider unavailable")),
            patch(
                "app.travel_brief.assess_travel_concern",
                return_value={
                    "risk_level": "low",
                    "final": "Cebu travel conditions could not be fully assessed from the currently gathered data.",
                    "travel_advice": ["Local news context could not be confirmed from the current scan."],
                    "weather_reasons": ["mainly clear weather supports routine plans"],
                    "news_reasons": [],
                    "relevant_news_items": [],
                },
            ),
        ):
            brief, err = build_travel_brief("Cebu")

        self.assertEqual(err, "provider unavailable")
        self.assertIn("could not be fully assessed", brief["final"].lower())
        self.assertFalse(
            any(
                "did not surface a major traveler-facing disruption" in item.lower() for item in brief["travel_advice"]
            ),
            brief["travel_advice"],
        )
        self.assertIn("Local news context could not be confirmed", brief["travel_advice"][0])


if __name__ == "__main__":
    unittest.main()
