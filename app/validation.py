from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, StringConstraints

MAX_PLACE_LENGTH = 100
MAX_QUESTION_LENGTH = 2_000
MAX_REQUEST_BODY_BYTES = 32_000


def strip_text(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


PlaceValue = Annotated[
    str,
    BeforeValidator(strip_text),
    StringConstraints(min_length=1, max_length=MAX_PLACE_LENGTH),
]

