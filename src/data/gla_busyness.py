"""GLA People Counts (O2 Motion) loader.

Real hourly London footfall at MSOA level from O2 mobile data, published by
the Greater London Authority under OGL v3.

Dataset: https://data.london.gov.uk/dataset/busyness-people-counts

MANUAL DOWNLOAD REQUIRED
    The London Datastore blocks programmatic access via Cloudflare.
    1. Open the dataset page in a browser
    2. Download the hourly People Counts CSV(s)
    3. Place them in data/gla_busyness/
    4. Run this module to load and filter to the target MSOA

MSOA E02000972 = Fitzrovia West & Soho (contains Wagamama Soho, 51.5131, -0.1318)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

DEFAULT_DATA_DIR = Path("data/gla_busyness")

# Target MSOA (2011 boundaries) for Wagamama Soho area
SOHO_MSOA_CODE = "E02000972"
SOHO_MSOA_NAME = "Fitzrovia West & Soho"
SOHO_LAT = 51.5131
SOHO_LNG = -0.1318

# The exact column names in the GLA CSV are not verifiable without opening
# the file in a browser. This flexible mapper handles likely variants.
_MSOA_COL_CANDIDATES = ["msoa_code", "msoa11cd", "MSOA_CODE", "MSOA11CD", "area_code"]
_TS_COL_CANDIDATES = ["datetime", "timestamp", "date_time", "hour", "date"]
_COUNT_COL_CANDIDATES = [
    "count", "people_count", "n_people", "resident_count", "worker_count",
]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _discover_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))


def load_gla_busyness(
    data_dir: Path | None = None,
    msoa_code: str = SOHO_MSOA_CODE,
) -> pd.DataFrame:
    """Load GLA People Counts and filter to a single MSOA.

    Returns a DataFrame with columns:
        timestamp_utc (datetime UTC)
        restaurant_id (str — the MSOA code used as the synthetic location id)
        cover_count (int — the footfall value, used as the demand target)

    The model will treat `cover_count` as the target variable, so downstream
    feature engineering and trainer code works unchanged.
    """
    d = data_dir or DEFAULT_DATA_DIR
    csv_files = _discover_csv_files(d)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {d}. Download the People Counts CSV "
            f"from https://data.london.gov.uk/dataset/busyness-people-counts "
            f"and place it in {d}/"
        )

    log.info("gla.load.start", files=len(csv_files), dir=str(d))

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        msoa_col = _find_column(df, _MSOA_COL_CANDIDATES)
        ts_col = _find_column(df, _TS_COL_CANDIDATES)
        if msoa_col is None or ts_col is None:
            log.warning(
                "gla.load.skip_file",
                path=str(path),
                reason="could not identify MSOA or timestamp columns",
                columns=list(df.columns),
            )
            continue

        df = df[df[msoa_col] == msoa_code].copy()
        if df.empty:
            continue
        df = df.rename(columns={ts_col: "timestamp_local"})
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No rows found for MSOA={msoa_code}. Check MSOA code or file contents."
        )

    combined = pd.concat(frames, ignore_index=True)

    combined["timestamp_local"] = pd.to_datetime(
        combined["timestamp_local"], errors="coerce", utc=False
    )
    combined = combined.dropna(subset=["timestamp_local"])

    combined["timestamp_utc"] = (
        combined["timestamp_local"]
        .dt.tz_localize("Europe/London", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    count_col = _find_column(combined, _COUNT_COL_CANDIDATES)
    if count_col is None:
        numeric_cols = combined.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise RuntimeError(
                f"No numeric count column found. Columns: {list(combined.columns)}"
            )
        count_col = numeric_cols[0]
        log.warning(
            "gla.load.inferred_count_col",
            using=count_col,
            candidates=_COUNT_COL_CANDIDATES,
        )

    result = pd.DataFrame(
        {
            "timestamp_utc": combined["timestamp_utc"].values,
            "local_ts": combined["timestamp_local"].values,
            "restaurant_id": msoa_code,
            "cover_count": combined[count_col].astype(float).round().astype(int),
            "channel": "dine_in",
        }
    )

    result = result.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    result = result.reset_index(drop=True)

    log.info(
        "gla.load.done",
        msoa=msoa_code,
        rows=len(result),
        date_range=f"{result['timestamp_utc'].min()} to {result['timestamp_utc'].max()}",
    )
    return result


def restaurant_meta_for_soho() -> pd.DataFrame:
    """Single-restaurant metadata for the simulated Wagamama Soho.

    seating_capacity and turnover are nominal — only used for censoring logic
    and are consistent with Wagamama Soho's actual size.
    """
    return pd.DataFrame(
        [
            {
                "restaurant_id": SOHO_MSOA_CODE,
                "name": "Simulated Wagamama Soho (MSOA-anchored)",
                "postcode": "W1D 4DL",
                "lat": SOHO_LAT,
                "lng": SOHO_LNG,
                "timezone": "Europe/London",
                "country_code": "GB",
                "seating_capacity": 100,
                "turnover_rate_per_hour": 1.75,
                "city_tier": "tier1",
                "footfall_zone_class": "very_high",
            }
        ]
    )
