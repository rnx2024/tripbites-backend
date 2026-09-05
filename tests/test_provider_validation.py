from __future__ import annotations

from app.provider_schemas import bounded_text, finite_coordinate
from app.routing.ors_service import _parse_route_payload


def test_coordinates_reject_nan_and_out_of_range_values() -> None:
    assert finite_coordinate(float("nan"), minimum=-90, maximum=90) is None
    assert finite_coordinate(91, minimum=-90, maximum=90) is None
    assert finite_coordinate(10.5, minimum=-90, maximum=90) == 10.5


def test_provider_text_is_bounded() -> None:
    assert bounded_text("  weather  ", maximum=20) == "weather"
    assert bounded_text("x" * 10, maximum=5) == "xxxxx"
    assert bounded_text(None, maximum=5) is None


def test_route_payload_rejects_malformed_or_negative_summary() -> None:
    assert _parse_route_payload({"features": []}, "driving-car") == (None, "no_route")
    assert _parse_route_payload({"features": [{"properties": {}}]}, "driving-car") == (None, "invalid_summary")
    assert _parse_route_payload(
        {"features": [{"properties": {"summary": {"distance": -1, "duration": 10}}}]},
        "driving-car",
    ) == (None, "invalid_summary")
