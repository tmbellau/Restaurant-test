"""POS CSV ingester.

Drop a CSV into ``data/pos_inbox/``, call ``ingest_inbox()``, and the ingester
will:

1. Validate every row against ``POSRowSchema`` (Pandera, blocking).
2. UPSERT into ``sales_records`` using (transaction_id, item_id) as the conflict
   key, with newer ``record_version`` winning.
3. Move the file to ``data/pos_processed/`` on success, ``data/pos_quarantine/``
   on validation failure.

Late-arriving corrections (voids, comps, split checks) come in as rows with the
same transaction_id but higher record_version and overwrite in place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandera as pa
import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import settings
from src.db.engine import SessionLocal
from src.signals.pos.schemas import POSRowSchema

log = structlog.get_logger(__name__)


def _quarantine_dir() -> Path:
    d = settings.pos_inbox_dir.parent / "pos_quarantine"
    d.mkdir(parents=True, exist_ok=True)
    return d


def validate_csv(path: Path) -> pd.DataFrame:
    """Read a POS CSV and validate against POSRowSchema. Raises on failure."""
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    validated: pd.DataFrame = POSRowSchema.validate(df, lazy=True)  # type: ignore[assignment]
    return validated


def upsert_rows(session: Session, df: pd.DataFrame) -> int:
    """UPSERT POS rows into sales_records. Returns count of rows written."""
    if df.empty:
        return 0

    rows = df.to_dict(orient="records")
    stmt = text(
        """
        INSERT INTO sales_records (
            restaurant_id, timestamp_utc, channel, transaction_id, item_id,
            item_qty, item_price, cover_count, voided, comped, record_version,
            ingested_at_utc
        ) VALUES (
            :restaurant_id, :timestamp_utc, :channel, :transaction_id, :item_id,
            :item_qty, :item_price, :cover_count, :voided, :comped, :record_version,
            :ingested_at_utc
        )
        ON CONFLICT (transaction_id, item_id)
        DO UPDATE SET
            restaurant_id = EXCLUDED.restaurant_id,
            timestamp_utc = EXCLUDED.timestamp_utc,
            channel = EXCLUDED.channel,
            item_qty = EXCLUDED.item_qty,
            item_price = EXCLUDED.item_price,
            cover_count = EXCLUDED.cover_count,
            voided = EXCLUDED.voided,
            comped = EXCLUDED.comped,
            record_version = EXCLUDED.record_version,
            ingested_at_utc = EXCLUDED.ingested_at_utc
        WHERE sales_records.record_version < EXCLUDED.record_version
        """
    )
    now = datetime.now(timezone.utc)
    for row in rows:
        row.setdefault("ingested_at_utc", now)
    session.execute(stmt, rows)
    session.commit()
    return len(rows)


def ingest_file(path: Path, session: Session | None = None) -> int:
    """Validate and upsert a single POS file. Returns rows written."""
    own_session = session is None
    if own_session:
        session = SessionLocal()
    try:
        df = validate_csv(path)
        count = upsert_rows(session, df)  # type: ignore[arg-type]
        processed = settings.pos_processed_dir
        processed.mkdir(parents=True, exist_ok=True)
        path.rename(processed / path.name)
        log.info("pos.ingest.success", file=path.name, rows=count)
        return count
    except pa.errors.SchemaErrors as exc:
        q = _quarantine_dir() / path.name
        path.rename(q)
        log.error(
            "pos.ingest.schema_error",
            file=path.name,
            quarantined_to=str(q),
            errors=str(exc.failure_cases.head(10)),
        )
        raise
    finally:
        if own_session and session is not None:
            session.close()


def ingest_inbox() -> dict[str, int]:
    """Process every CSV currently in the inbox. Returns {filename: rows_written}."""
    settings.pos_inbox_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for path in sorted(settings.pos_inbox_dir.glob("*.csv")):
        try:
            results[path.name] = ingest_file(path)
        except Exception as exc:  # noqa: BLE001
            results[path.name] = -1
            log.error("pos.ingest.file_failed", file=path.name, error=str(exc))
    return results
