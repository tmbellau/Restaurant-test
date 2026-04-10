"""Time zone handling — the single source of truth for all UTC <-> local conversions.

MANDATORY RULES (enforced by code review):

1. All timestamps stored in the database are UTC. Column names end in `_utc`.
2. Feature computation happens in LOCAL time. "Same hour last week" means local
   hour, not UTC hour — critical for DST boundaries.
3. The feature assembler converts UTC -> local at the boundary, computes features,
   and outputs UTC timestamps for storage.
4. DST transitions must be unit-tested for every feature that depends on "same hour
   last week" or similar local-time lookups.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


class TimeZoneManager:
    """Handles all UTC <-> local conversions for a given IANA time zone name.

    Uses stdlib ``zoneinfo`` (Python 3.9+) which correctly handles DST
    transitions and historical tz data.
    """

    def __init__(self, tz_name: str) -> None:
        self.tz_name = tz_name
        self._tz = ZoneInfo(tz_name)

    @property
    def zone(self) -> ZoneInfo:
        return self._tz

    # ---- conversions ----

    def to_local(self, dt_utc: datetime) -> datetime:
        """Convert a UTC datetime to this zone's local time.

        If ``dt_utc`` is naive, it is assumed to be UTC.
        """
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(self._tz)

    def to_utc(self, dt_local: datetime) -> datetime:
        """Convert a local datetime to UTC.

        If ``dt_local`` is naive, it is assumed to be in this zone.
        """
        if dt_local.tzinfo is None:
            dt_local = dt_local.replace(tzinfo=self._tz)
        return dt_local.astimezone(timezone.utc)

    # ---- feature helpers ----

    def local_hour(self, dt_utc: datetime) -> int:
        """Local hour-of-day (0-23) for a given UTC timestamp."""
        return self.to_local(dt_utc).hour

    def local_dow(self, dt_utc: datetime) -> int:
        """Local day-of-week (Mon=0 .. Sun=6) for a given UTC timestamp."""
        return self.to_local(dt_utc).weekday()

    def same_local_hour_days_ago(self, dt_utc: datetime, days: int) -> datetime:
        """Return the UTC timestamp that has the same local wall-clock hour
        ``days`` days earlier.

        This is DST-aware: if we go from 2024-10-29 17:00 BST to 7 days earlier,
        the result is 2024-10-22 17:00 BST -- i.e. the same local 17:00, not
        the UTC equivalent.

        On DST transition days the target local hour may be ambiguous or
        non-existent; we fall back to the closest valid local time.
        """
        local = self.to_local(dt_utc)
        target_local = local - timedelta(days=days)
        # Re-attach zone explicitly to re-resolve DST at the target date.
        naive = target_local.replace(tzinfo=None)
        target_localised = naive.replace(tzinfo=self._tz)
        return target_localised.astimezone(timezone.utc)
