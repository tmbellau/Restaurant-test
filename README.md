# Wagamama Forecast

Demand forecasting platform for walk-in-heavy restaurant chains. Predicts
dine-in covers and delivery orders at hourly and daily granularity with
calibrated confidence intervals, and produces staffing + purchasing
recommendations.

See the full build spec in the plan file (`/root/.claude/plans/bubbly-wibbling-pillow.md`).

## Current status

**Layer 1 (foundation) -- complete:**

- Project scaffold, config via Pydantic Settings, Docker Compose for TimescaleDB + Redis
- `src/util/time_zones.py` -- UTC <-> local conversion with DST-safe `same_local_hour_days_ago`
- SQLAlchemy models + Alembic migration `0001_initial_layer1`
  - `restaurants`, `sales_records`, `signal_features`, `predictions`, `location_closures`
  - TimescaleDB hypertables on `sales_records` and `signal_features`
- Signal hub (`src/signals/base.py`, `src/signals/registry.py`)
  - `SignalSource` Protocol + `BaseSignalSource` ABC with Tenacity retries + Pybreaker circuit breaker
  - `SignalRegistry` auto-discovers sources via `pkgutil`
- POS ingestion (`src/signals/pos/`)
  - CSV ingester with Pandera validation, quarantine on failure
  - Late-correction UPSERT using `(transaction_id, item_id)` as dedup key
  - `ZonalAdapter` stub for Layer 2 API work
- Weather source: `OpenMeteoWeatherSource` (free, no key required)
- Unit tests covering DST transitions, POS schema validation, signal registry discovery, retry/breaker wrapping
- `scripts/verify_layer1.py` end-to-end verification script

## Layer 1 verification gate

```bash
# 1. Bring up TimescaleDB
docker compose up -d db

# 2. Install deps
pip install -e ".[dev,ml]"

# 3. Run migrations
alembic upgrade head

# 4. Run unit tests
pytest tests/test_features/test_time_zones.py tests/test_signals/

# 5. Run the end-to-end verification script
python scripts/verify_layer1.py
```

## Layers remaining

- **Layer 2** -- core pipeline: feature assembler, censoring, LightGBM trainer, CQR, SHAP,
  batch prediction, FastAPI read endpoints, minimal dashboard, demo data generator, Dagster
- **Layer 3** -- production hardening: delivery channel, promos, manager overrides, business
  metrics, drift detection, temporal reconciliation, GDPR, CI/CD
- **Layer 4** -- advanced: ingredient BOM, religious calendar, social sentiment plug-in,
  economic indicators, multi-region

## Project layout

```
src/
  config.py                  # Pydantic Settings
  util/time_zones.py         # DST-safe UTC<->local
  db/engine.py               # SQLAlchemy engine + session
  db/models.py               # ORM models
  signals/
    base.py                  # SignalSource Protocol + BaseSignalSource ABC
    registry.py              # Auto-discovery
    weather.py               # Open-Meteo
    pos/
      schemas.py             # Pandera row schema
      csv_ingester.py        # CSV ingester with UPSERT
      base_adapter.py        # POS API adapter ABC
      zonal_adapter.py       # Zonal POS stub
tests/
  test_features/test_time_zones.py
  test_signals/test_layer1_gate.py
alembic/versions/0001_initial_layer1.py
scripts/verify_layer1.py
```
