"""TfL Santander Cycles trip data → daily trip count target for Soho.

Data source: https://cycling.data.tfl.gov.uk/usage-stats/ (TfL Open Data, OGLv2)

Each weekly CSV contains per-trip records with start/end timestamp and station.
We filter to Soho-area docking stations and aggregate trip starts+ends to
per-date counts. Target: daily Santander trips at Soho stations — a real,
freely-available, location-specific demand signal driven by weather,
holidays, events and temporal patterns.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

S3_BUCKET = "https://s3-eu-west-1.amazonaws.com/cycling.data.tfl.gov.uk"
DOWNLOAD_BASE = "https://cycling.data.tfl.gov.uk"
CACHE_DIR = Path("data/santander")

# Soho-area docking stations (verified present in 2022 data). Station numbers
# are stable identifiers; names can have minor variants (spaces etc.).
SOHO_STATIONS: dict[str, str] = {
    "003504": "Moor Street, Soho",
    "001163": "Wardour Street, Soho",
    "003489": "Broadwick Street, Soho",
}

# Additional nearby stations (Covent Garden / West End) — within ~500m of Soho
NEARBY_STATIONS: dict[str, str] = {
    "001229": "High Holborn, Covent Garden",
    "003452": "Panton Street, West End",
    "001067": "St. James's Square, St. James's",
    "300056": "Golden Square, Soho",
}

LOCATION_ID = "soho_cycles"
SOHO_LAT = 51.5131
SOHO_LNG = -0.1318

# Kept as legacy aliases — many downstream modules import these names.
RESTAURANT_ID = LOCATION_ID
WAGAMAMA_SOHO_LAT = SOHO_LAT
WAGAMAMA_SOHO_LNG = SOHO_LNG


def list_weekly_files(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[tuple[str, datetime, datetime]]:
    """List Santander weekly CSV files via S3 XML listing.

    Returns list of (s3_key, start_date, end_date) tuples, filtered to files
    whose date range overlaps [start_date, end_date].
    """
    files: list[tuple[str, datetime, datetime]] = []
    continuation: str | None = None
    page = 0

    while True:
        params: dict[str, str] = {"list-type": "2", "prefix": "usage-stats/"}
        if continuation:
            params["continuation-token"] = continuation

        import time
        xml = None
        for attempt in range(5):
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.get(S3_BUCKET + "/", params=params)
                    resp.raise_for_status()
                    xml = resp.text
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (503, 502, 504) and attempt < 4:
                    wait = 2 ** attempt
                    log.info("santander.list.retry", attempt=attempt, wait=wait)
                    time.sleep(wait)
                    continue
                raise
        if xml is None:
            raise RuntimeError("Failed to list S3 bucket after retries")

        root = ET.fromstring(xml)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        for content in root.findall("s3:Contents", ns):
            key = content.find("s3:Key", ns).text
            if not key or not key.endswith(".csv"):
                continue
            date_range = _parse_filename_dates(key)
            if date_range is None:
                continue
            file_start, file_end = date_range
            if start_date and file_end < start_date:
                continue
            if end_date and file_start > end_date:
                continue
            files.append((key, file_start, file_end))

        next_token = root.find("s3:NextContinuationToken", ns)
        if next_token is None or not next_token.text:
            break
        continuation = next_token.text
        page += 1
        if page > 20:
            log.warning("santander.list.max_pages_reached")
            break

    files.sort(key=lambda x: x[1])
    log.info("santander.list.done", n_files=len(files))
    return files


_DATE_PATTERNS = [
    re.compile(r"(\d{1,2})([A-Za-z]{3})(\d{2,4})-(\d{1,2})([A-Za-z]{3})(\d{2,4})"),
]


def _parse_filename_dates(filename: str) -> tuple[datetime, datetime] | None:
    """Parse date range from filename like '347JourneyDataExtract05Dec2022-11Dec2022.csv'."""
    cleaned = filename.replace(" ", "")
    for pattern in _DATE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        try:
            d1, m1, y1, d2, m2, y2 = match.groups()
            y1 = f"20{y1}" if len(y1) == 2 else y1
            y2 = f"20{y2}" if len(y2) == 2 else y2
            start = datetime.strptime(f"{d1} {m1} {y1}", "%d %b %Y")
            end = datetime.strptime(f"{d2} {m2} {y2}", "%d %b %Y")
            return start, end
        except ValueError:
            continue
    return None


def download_and_filter(
    s3_key: str,
    station_ids: set[str],
    cache_dir: Path | None = None,
    station_names: set[str] | None = None,
) -> pd.DataFrame:
    """Download a weekly CSV, filter to target stations, return trip rows.

    Only keeps trips where EITHER start OR end station is in our set,
    since both contribute to area footfall. Matches by station ID *or*
    station name (for compatibility with pre-2022 CSVs whose IDs differ).
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(s3_key).name.replace(" ", "_")
    filtered_path = cache_dir / f"filtered_{fname}"
    if filtered_path.exists():
        df = pd.read_csv(filtered_path)
        if "start_ts" in df.columns:
            df["start_ts"] = pd.to_datetime(df["start_ts"], errors="coerce")
        if "end_ts" in df.columns:
            df["end_ts"] = pd.to_datetime(df["end_ts"], errors="coerce")
        if "start_station" in df.columns:
            df["start_station"] = df["start_station"].astype(str).str.zfill(6)
        if "end_station" in df.columns:
            df["end_station"] = df["end_station"].astype(str).str.zfill(6)
        return df

    url = f"{DOWNLOAD_BASE}/{s3_key}"
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        raw = resp.content

    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    start_col = _find_col(df, ["Start station number", "StartStation Id", "Start Station Id"])
    end_col = _find_col(df, ["End station number", "EndStation Id", "End Station Id"])
    start_name_col = _find_col(df, ["Start station", "StartStation Name", "Start Station Name"])
    end_name_col = _find_col(df, ["End station", "EndStation Name", "End Station Name"])
    start_ts = _find_col(df, ["Start date", "Start Date"])
    end_ts = _find_col(df, ["End date", "End Date"])

    if not all([start_col, end_col, start_ts, end_ts]):
        log.warning("santander.file.skip", key=s3_key, cols=list(df.columns)[:8])
        return pd.DataFrame()

    # Station IDs: 2022+ uses zero-padded 6-digit; 2019 uses plain integers.
    # Station NAMES are stable across both eras — filter by name.
    df[start_col] = df[start_col].astype(str).str.strip().str.zfill(6)
    df[end_col] = df[end_col].astype(str).str.strip().str.zfill(6)

    mask = df[start_col].isin(station_ids) | df[end_col].isin(station_ids)

    if start_name_col and end_name_col and station_names:
        # Fallback/augment: match by normalised station name
        def _norm(s: object) -> str:
            return "".join(str(s).lower().split()).replace(",", "")
        start_norm = df[start_name_col].map(_norm)
        end_norm = df[end_name_col].map(_norm)
        normset = {_norm(n) for n in station_names}
        mask = mask | start_norm.isin(normset) | end_norm.isin(normset)

    filtered = df[mask].copy()

    # Parse timestamps. Different eras use different formats; try ISO first
    # then fall back to dayfirst dd/mm/yyyy.
    def _parse_ts(series: pd.Series) -> pd.Series:
        s1 = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d %H:%M")
        mask_nan = s1.isna()
        if mask_nan.any():
            s2 = pd.to_datetime(series.where(mask_nan), errors="coerce", dayfirst=True)
            s1 = s1.fillna(s2)
        return s1

    out = pd.DataFrame({
        "start_ts": _parse_ts(filtered[start_ts]),
        "end_ts": _parse_ts(filtered[end_ts]),
        "start_station": filtered[start_col],
        "end_station": filtered[end_col],
    }).dropna(subset=["start_ts"])

    out.to_csv(filtered_path, index=False)
    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def aggregate_hourly(
    trips: pd.DataFrame,
    station_ids: set[str],
    restaurant_id: str = RESTAURANT_ID,
) -> pd.DataFrame:
    """Aggregate per-trip records to hourly counts (starts + ends) at target stations.

    Returns columns: timestamp_utc, restaurant_id, cover_count, channel.
    """
    if trips.empty:
        return pd.DataFrame(columns=["timestamp_utc", "restaurant_id", "cover_count", "channel"])

    starts_mask = trips["start_station"].isin(station_ids)
    ends_mask = trips["end_station"].isin(station_ids)

    start_hours = (
        trips.loc[starts_mask, "start_ts"].dt.floor("h").value_counts().sort_index()
    )
    end_hours = (
        trips.loc[ends_mask, "end_ts"].dt.floor("h").value_counts().sort_index()
    )
    combined = start_hours.add(end_hours, fill_value=0).astype(int)

    idx = pd.to_datetime(combined.index)
    # Bike data timestamps are local London time; localize then convert to UTC.
    # ambiguous=False maps DST fall-back hours to standard time (consistent
    # handling). nonexistent="shift_forward" handles spring-forward gaps.
    idx_local = idx.tz_localize(
        "Europe/London", ambiguous=False, nonexistent="shift_forward"
    )
    idx_utc = idx_local.tz_convert("UTC")

    df = pd.DataFrame({
        "timestamp_utc": idx_utc,
        "restaurant_id": restaurant_id,
        "cover_count": combined.values,
        "channel": "dine_in",
    })
    return df


def load_santander_hourly(
    start_date: datetime,
    end_date: datetime,
    include_nearby: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """High-level: download + filter + aggregate Santander data for date range.

    Returns hourly DataFrame: timestamp_utc, restaurant_id, cover_count, channel.
    """
    station_ids = set(SOHO_STATIONS.keys())
    station_names = set(SOHO_STATIONS.values())
    if include_nearby:
        station_ids |= set(NEARBY_STATIONS.keys())
        station_names |= set(NEARBY_STATIONS.values())

    log.info(
        "santander.load.start",
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        n_stations=len(station_ids),
    )

    files = list_weekly_files(start_date, end_date)
    if not files:
        raise RuntimeError("No Santander files found for date range")

    all_trips: list[pd.DataFrame] = []
    for i, (key, fs, fe) in enumerate(files, 1):
        log.info(
            "santander.download",
            file=f"{i}/{len(files)}",
            key=Path(key).name,
            range=f"{fs.date()}-{fe.date()}",
        )
        try:
            df = download_and_filter(key, station_ids, cache_dir=cache_dir, station_names=station_names)
            if not df.empty:
                all_trips.append(df)
        except Exception as e:
            log.warning("santander.download.error", key=key, error=str(e))

    if not all_trips:
        raise RuntimeError("No trips found at target stations")

    trips = pd.concat(all_trips, ignore_index=True)
    trips = trips[
        (trips["start_ts"] >= start_date) & (trips["start_ts"] <= end_date)
    ].reset_index(drop=True)

    log.info("santander.trips.total", n_trips=len(trips))

    hourly = aggregate_hourly(trips, station_ids)

    # Clamp to the requested date range in UTC; phantom timestamps (e.g. from
    # record-level timestamp corruption) are dropped here.
    start_utc = pd.Timestamp(start_date, tz="UTC")
    end_utc = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    if not hourly.empty:
        hourly = hourly[
            (hourly["timestamp_utc"] >= start_utc)
            & (hourly["timestamp_utc"] <= end_utc)
        ].copy()

    # Ensure continuous hourly index over the requested range (fill missing with 0)
    if not hourly.empty:
        ts_min = hourly["timestamp_utc"].min().floor("h")
        ts_max = hourly["timestamp_utc"].max().floor("h")
        full_idx = pd.date_range(ts_min, ts_max, freq="h", tz="UTC")
        hourly = hourly.set_index("timestamp_utc").reindex(full_idx)
        hourly["cover_count"] = hourly["cover_count"].fillna(0).astype(int)
        hourly["restaurant_id"] = RESTAURANT_ID
        hourly["channel"] = "dine_in"
        hourly = hourly.reset_index().rename(columns={"index": "timestamp_utc"})

    log.info(
        "santander.hourly.done",
        n_hours=len(hourly),
        date_range=f"{hourly['timestamp_utc'].min()} to {hourly['timestamp_utc'].max()}",
        total_trips_scaled=int(hourly["cover_count"].sum()) if not hourly.empty else 0,
    )
    return hourly


def location_meta() -> pd.DataFrame:
    """Metadata for the target location (Soho Santander docking area)."""
    return pd.DataFrame([{
        "restaurant_id": LOCATION_ID,  # column name kept for pipeline compatibility
        "name": "Soho Cycles (aggregate of Soho/West End stations)",
        "lat": SOHO_LAT,
        "lng": SOHO_LNG,
        "timezone": "Europe/London",
        "country_code": "GB",
        # Below are inert pass-through fields required by the feature assembler.
        # They carry no information for this target (single location).
        "seating_capacity": 0,
        "turnover_rate_per_hour": 0,
        "city_tier": "tier1",
        "footfall_zone_class": "very_high",
    }])


# Legacy alias.
restaurant_meta = location_meta
