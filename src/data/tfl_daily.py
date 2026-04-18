"""TfL daily station entry/exit data loader.

Sources (try in order):
    1. Local CSV in data/tfl/ (user-downloaded)
    2. Kaggle dataset: kaggle.com/datasets/olisao/transport-for-london-tfl-entry-and-exit-dataset

The TAPS (Ticketing and Passenger System) data provides daily entries and
exits per station. Covers LU, LO, DLR, TfL Rail from 2019 onward.

We use Tottenham Court Road (closest Tube to Wagamama Soho at ~400m).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

DEFAULT_DATA_DIR = Path("data/tfl")

# Station we want — Tottenham Court Road is ~400m from Wagamama Soho.
# Other options: Leicester Square (~450m), Piccadilly Circus (~500m).
TARGET_STATION = "Tottenham Court Road"

# Flexible column name matching for TfL CSV variants.
_STATION_COL_CANDIDATES = [
    "station", "station_name", "Station", "StationName",
    "stn", "Station Name", "nlc_desc",
]
_DATE_COL_CANDIDATES = [
    "date", "Date", "day", "Day", "calendar_date",
]
_ENTRIES_COL_CANDIDATES = [
    "entries", "Entries", "entry", "Entry", "EntryCount",
    "total_entries", "tap_in",
]
_EXITS_COL_CANDIDATES = [
    "exits", "Exits", "exit", "Exit", "ExitCount",
    "total_exits", "tap_out",
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_tfl_daily(
    data_dir: Path | None = None,
    station_name: str = TARGET_STATION,
) -> pd.DataFrame:
    """Load TfL daily station data and filter to a specific station.

    Returns DataFrame with: date, entries, exits, total_footfall.
    """
    d = data_dir or DEFAULT_DATA_DIR
    csv_files = sorted(d.glob("*.csv")) if d.exists() else []
    if not csv_files:
        raise FileNotFoundError(
            f"No TfL CSV files found in {d}. Download from one of:\n"
            f"  1. TfL transparency page: search 'TAPS daily rail station entry exit'\n"
            f"  2. Kaggle: kaggle.com/datasets/olisao/transport-for-london-tfl-entry-and-exit-dataset\n"
            f"  3. TfL Open Data: tfl.gov.uk/info-for/open-data-users/our-open-data\n"
            f"Place CSV files in {d}/"
        )

    log.info("tfl.load.start", files=len(csv_files), dir=str(d))

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            log.warning("tfl.load.skip_file", path=str(path), error=str(e))
            continue

        station_col = _find_col(df, _STATION_COL_CANDIDATES)
        date_col = _find_col(df, _DATE_COL_CANDIDATES)
        entries_col = _find_col(df, _ENTRIES_COL_CANDIDATES)
        exits_col = _find_col(df, _EXITS_COL_CANDIDATES)

        if station_col is None or date_col is None:
            log.warning(
                "tfl.load.skip_file",
                path=str(path),
                reason="could not identify station or date columns",
                columns=list(df.columns)[:10],
            )
            continue

        # Filter to target station (case-insensitive substring match)
        mask = df[station_col].astype(str).str.contains(
            station_name, case=False, na=False
        )
        station_df = df[mask].copy()
        if station_df.empty:
            continue

        station_df["date"] = pd.to_datetime(station_df[date_col], errors="coerce")
        station_df = station_df.dropna(subset=["date"])

        # Extract entries and exits, defaulting to 0 if not found
        if entries_col:
            station_df["entries"] = pd.to_numeric(
                station_df[entries_col], errors="coerce"
            ).fillna(0)
        else:
            station_df["entries"] = 0

        if exits_col:
            station_df["exits"] = pd.to_numeric(
                station_df[exits_col], errors="coerce"
            ).fillna(0)
        else:
            station_df["exits"] = 0

        frames.append(station_df[["date", "entries", "exits"]])

    if not frames:
        available_stations = _list_available_stations(csv_files)
        raise RuntimeError(
            f"No rows found for station '{station_name}'. "
            f"Available stations (sample): {available_stations[:10]}"
        )

    result = pd.concat(frames, ignore_index=True)

    # Aggregate to one row per date (in case of duplicates)
    result = result.groupby("date", as_index=False).agg(
        {"entries": "sum", "exits": "sum"}
    )
    result["total_footfall"] = result["entries"] + result["exits"]
    result = result.sort_values("date").reset_index(drop=True)

    log.info(
        "tfl.load.done",
        station=station_name,
        rows=len(result),
        date_range=f"{result['date'].min().date()} to {result['date'].max().date()}",
        avg_daily_footfall=int(result["total_footfall"].mean()),
    )
    return result


def _list_available_stations(csv_files: list[Path]) -> list[str]:
    """Helper to list available stations for error messages."""
    stations: set[str] = set()
    for path in csv_files[:3]:
        try:
            df = pd.read_csv(path, nrows=10000, low_memory=False)
            col = _find_col(df, _STATION_COL_CANDIDATES)
            if col:
                stations.update(df[col].dropna().unique()[:20])
        except Exception:
            pass
    return sorted(stations)[:20]
