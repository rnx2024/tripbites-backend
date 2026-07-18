# app/tooling/retry_policy.py
from __future__ import annotations

import logging

from retryguard import ErrorClassifier
from retryguard.integrations.tenacity import (
    before_sleep_log_retryguard,
    retry_if_retryguard,
    wait_retryguard,
)
from tenacity import retry, stop_after_attempt

log = logging.getLogger(__name__)

RETRY_MAX_ATTEMPTS = 3
RETRY_FALLBACK_DELAY_SECONDS = 0.5

classifier = ErrorClassifier()


def build_http_retry(
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    fallback_seconds: float = RETRY_FALLBACK_DELAY_SECONDS,
):
    """
    Tenacity retry decorator classified by retryguard: retries only exceptions
    retryguard marks retryable (timeouts, 429/5xx, network errors), honoring
    Retry-After when present. Reraises the original exception so callers keep
    their existing except-block error handling.
    """
    return retry(
        retry=retry_if_retryguard(classifier),
        wait=wait_retryguard(classifier, fallback_seconds=fallback_seconds),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log_retryguard(log, classifier=classifier),
        reraise=True,
    )
