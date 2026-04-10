"""Signal hub core: SignalSource Protocol and BaseSignalSource ABC.

Every external data source that feeds features into the prediction pipeline
implements SignalSource. New sources drop into src/signals/ and are picked up
automatically by SignalRegistry on the next Dagster run (Layer 2).

Production concerns baked into BaseSignalSource:
    - Retries with exponential backoff (Tenacity)
    - Circuit breaker per source (Pybreaker) so one flaky API doesn't cascade
    - Structured logging with source_name context
    - Freshness tracking via last_successful_fetch_utc
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import pybreaker
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)


@runtime_checkable
class SignalSource(Protocol):
    """The contract every signal source must fulfil."""

    source_name: str
    refresh_interval_seconds: int
    feature_columns: list[str]

    def fetch_raw(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame: ...

    def transform_to_features(self, raw_df: pd.DataFrame) -> pd.DataFrame: ...

    def get_feature_schema(self) -> dict[str, str]: ...


class SignalFetchError(Exception):
    """Raised when a signal source fails to fetch data after all retries."""


class BaseSignalSource(abc.ABC):
    """Production-grade base class for signal sources.

    Subclasses implement `_fetch_raw_impl`, `transform_to_features`, and
    `get_feature_schema`. This base class wraps `_fetch_raw_impl` with retry
    and circuit-breaker logic and exposes `fetch_raw` as the public entrypoint.
    """

    source_name: str = "base"
    refresh_interval_seconds: int = 3600
    feature_columns: list[str] = []

    # Default circuit breaker: open after 5 failures, stay open for 60s.
    _breaker_fail_max: int = 5
    _breaker_reset_timeout: int = 60

    def __init__(self) -> None:
        self.last_successful_fetch_utc: datetime | None = None
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=self._breaker_fail_max,
            reset_timeout=self._breaker_reset_timeout,
            name=f"signal:{self.source_name}",
        )

    # ---- public API ----

    def fetch_raw(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        """Fetch raw signal data with retry + circuit-breaker protection."""
        try:
            df = self._retry_wrapper(restaurant_id, start_utc, end_utc)
            self.last_successful_fetch_utc = datetime.now(timezone.utc)
            log.info(
                "signal.fetch.success",
                source=self.source_name,
                restaurant_id=restaurant_id,
                rows=len(df),
            )
            return df
        except pybreaker.CircuitBreakerError as exc:
            log.error("signal.circuit_open", source=self.source_name, error=str(exc))
            raise SignalFetchError(
                f"Circuit breaker open for {self.source_name}"
            ) from exc
        except Exception as exc:
            log.error(
                "signal.fetch.failed",
                source=self.source_name,
                restaurant_id=restaurant_id,
                error=str(exc),
            )
            raise SignalFetchError(f"{self.source_name} fetch failed: {exc}") from exc

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OSError, ConnectionError, TimeoutError)),
    )
    def _retry_wrapper(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        return self._breaker.call(
            self._fetch_raw_impl, restaurant_id, start_utc, end_utc
        )

    # ---- subclass hooks ----

    @abc.abstractmethod
    def _fetch_raw_impl(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        """Actually fetch the data. Subclasses implement. Raise on failure."""

    @abc.abstractmethod
    def transform_to_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw fetched data into the feature columns this source emits."""

    @abc.abstractmethod
    def get_feature_schema(self) -> dict[str, str]:
        """Return a mapping of feature_name -> dtype (e.g. 'float', 'int', 'bool')."""

    # ---- metadata ----

    def health(self) -> dict[str, Any]:
        """Health snapshot for freshness monitoring."""
        return {
            "source": self.source_name,
            "last_successful_fetch_utc": (
                self.last_successful_fetch_utc.isoformat()
                if self.last_successful_fetch_utc
                else None
            ),
            "circuit_state": self._breaker.current_state,
            "refresh_interval_seconds": self.refresh_interval_seconds,
        }
