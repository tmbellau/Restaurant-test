"""Compare stored predictions against actuals once they arrive.

Each time predict_future.py runs, it drops a parquet in data/predictions/.
This script loads ALL of them, joins against the latest daily actuals,
reports accuracy per prediction batch + per horizon, and saves a
consolidated track-record parquet.

Usage:
    python scripts/compare_predictions.py
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_DIR = Path("data/predictions")
TRACK_RECORD = PREDICTIONS_DIR / "track_record.parquet"


def main() -> None:
    if not PREDICTIONS_DIR.exists():
        print(f"No predictions directory at {PREDICTIONS_DIR}")
        return

    daily = pd.read_parquet("data/training/daily_training_features.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    actuals: dict[date, float] = {
        d.date(): float(c) for d, c in zip(daily["date"], daily["cover_count"])
    }

    # Collect all prediction files (exclude the track record itself)
    files = [f for f in sorted(PREDICTIONS_DIR.glob("predictions_*.parquet"))]
    if not files:
        print("No prediction files to compare.")
        return

    all_preds = []
    for f in files:
        df = pd.read_parquet(f)
        df["prediction_file"] = f.name
        all_preds.append(df)

    preds = pd.concat(all_preds, ignore_index=True)
    preds["target"] = pd.to_datetime(preds["target"]).dt.date
    preds["origin"] = pd.to_datetime(preds["origin"]).dt.date

    # Attach actuals where available
    preds["actual"] = preds["target"].map(actuals)
    preds["error"] = preds["predicted"] - preds["actual"]
    preds["abs_error"] = preds["error"].abs()
    preds["pct_error"] = (preds["error"] / preds["actual"].clip(lower=1) * 100)

    # Scored = has an actual AND not a suspected outage day. Outages are
    # filtered so a broken station-data day doesn't tank the reported WAPE.
    has_actual = preds.dropna(subset=["actual"])
    outage_mask = has_actual["actual"] < 50
    scored = has_actual[~outage_mask]
    outages = has_actual[outage_mask]
    pending = preds[preds["actual"].isna()]

    # Always save the full track record
    preds.to_parquet(TRACK_RECORD, index=False)

    print("=" * 78)
    print(f"PREDICTION TRACK RECORD — {len(preds)} predictions across {len(files)} runs")
    print(f"Scored: {len(scored)}  Pending: {len(pending)}  Excluded outages: {len(outages)}")
    print("=" * 78)

    if len(scored) > 0:
        print("\nAccuracy by horizon:")
        print(f"  {'h':>3s} {'n':>5s} {'MAE':>8s} {'WAPE':>8s} {'med%':>7s}")
        for h in sorted(scored["horizon"].unique()):
            seg = scored[scored["horizon"] == h]
            mae = seg["abs_error"].mean()
            wape = seg["abs_error"].sum() / max(seg["actual"].abs().sum(), 1e-8)
            medp = seg["pct_error"].abs().median()
            print(f"  {int(h):>3d} {len(seg):>5d} {mae:>8.1f} {wape:>8.1%} {medp:>6.1f}%")

        print("\nPer-origin performance:")
        for origin, g in scored.groupby("origin"):
            wape = g["abs_error"].sum() / max(g["actual"].abs().sum(), 1e-8)
            print(f"  origin {origin}: n={len(g)}, WAPE={wape:.1%}")

    if len(pending) > 0:
        print(f"\nPending predictions ({len(pending)} rows):")
        show = pending[["generated_at", "origin", "target", "horizon", "predicted"]].head(15)
        print(show.to_string(index=False))
        if len(pending) > 15:
            print(f"  ... and {len(pending) - 15} more")

    print(f"\nTrack record saved: {TRACK_RECORD}")


if __name__ == "__main__":
    main()
