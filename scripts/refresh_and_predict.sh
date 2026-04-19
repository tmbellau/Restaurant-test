#!/usr/bin/env bash
# Daily refresh + forward prediction.
#
# 1. Pull any new Santander weekly CSVs from TfL's S3 bucket.
# 2. Rebuild the hourly + daily feature parquets.
# 3. Generate a 7-day-ahead forecast from the latest known actual.
# 4. Recompute prediction track record (scores previous runs against
#    any actuals that have arrived since).
#
# Schedule this via cron / a task runner for a genuine live system.
#
# Usage:   bash scripts/refresh_and_predict.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] Pulling any new Santander data..."
python -m src.data.build_training_set \
    --start 2017-01-01 \
    --end "$(date +%Y-%m-%d)"

echo ""
echo "[2/4] Rebuilding daily feature parquet..."
python -m src.data.build_daily_training_set --exclude-covid

echo ""
echo "[3/4] Generating 7-day-ahead forecast..."
python scripts/predict_future.py --days 7

echo ""
echo "[4/4] Scoring stored predictions against any new actuals..."
python scripts/compare_predictions.py

echo ""
echo "Done."
