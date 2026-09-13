import unittest
from unittest.mock import Mock, patch

from app.weather.openmeteo_provider import geocode_place


@patch("time.sleep", return_value=None)
class OpenMeteoGeocodingTests(unittest.TestCase):
    def test_geocode_place_normalizes_location(self, _sleep) -> None:
        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "name": "Cebu City",
                    "country": "Philippines",
                    "latitude": 10.3157,
                    "longitude": 123.8854,
                    "timezone": "Asia/Manila",
                }
            ]
        }
        response.raise_for_status.return_value = None

        with patch("app.weather.openmeteo_provider.httpx.get", return_value=response) as get_mock:
            location, error = geocode_place("Cebu City")

        self.assertIsNone(error)
        self.assertEqual(location["name"], "Cebu City")
        self.assertEqual(location["country"], "Philippines")
        self.assertEqual(location["timezone"], "Asia/Manila")
        get_mock.assert_called_once()

    def test_geocode_place_rejects_malformed_response(self, _sleep) -> None:
        response = Mock()
        response.json.return_value = {"results": [{"name": "Cebu City"}]}
        response.raise_for_status.return_value = None

        with patch("app.weather.openmeteo_provider.httpx.get", return_value=response):
            location, error = geocode_place("Cebu City")

        self.assertIsNone(location)
        self.assertEqual(error, "Invalid geocoding result.")


if __name__ == "__main__":
    unittest.main()
