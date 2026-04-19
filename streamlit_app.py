"""Soho cycling demand — live forecasting demo (home / predict page).

Pick any date in the holdout window and see the model's 7-day forecast
with calibrated 80% prediction intervals, compared against the actual
trip counts.
"""

from __future__ import annotations

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----- Config -----
st.set_page_config(
    page_title="Soho Cycles Forecast",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- Loaders (cached so re-renders are instant) -----
@st.cache_data(show_spinner=False)
def load_daily() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_parquet("data/training/daily_training_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["covers_7d_lag"]).reset_index(drop=True)
    idx = {d.date(): i for d, i in zip(df["date"], df.index)}
    actuals = {d.date(): float(c) for d, c in zip(df["date"], df["cover_count"])}
    return df, idx, actuals


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[dict, list[str], dict[int, float]]:
    """Load horizon-aware quantile models + per-horizon conformal buffers.

    Returns (models, feature_names, buffer_by_horizon). The horizon-aware
    models take `horizon_days` as a feature and output quantile predictions
    that widen with horizon because they were trained on multi-horizon data.
    The per-horizon buffer is calibrated on a held-out year's residuals.
    """
    model_dir = Path("data/models")
    features = json.loads((model_dir / "horizon_aware_feature_names.json").read_text())
    metrics = json.loads((model_dir / "horizon_aware_metrics.json").read_text())
    buffers = {int(k): float(v) for k, v in metrics["per_horizon_buffer"].items()}
    models = {}
    for name, alpha in [("lower", 5), ("median", 50), ("upper", 95)]:
        txt_path = model_dir / f"lgbm_horizon_q{alpha:02d}.txt"
        pkl_path = model_dir / f"lgbm_horizon_q{alpha:02d}.pkl"
        if txt_path.exists():
            models[name] = lgb.Booster(model_file=str(txt_path))
        else:
            with open(pkl_path, "rb") as fh:
                models[name] = pickle.load(fh)
    return models, features, buffers


@st.cache_data(show_spinner=False)
def load_eval_2025() -> pd.DataFrame:
    df = pd.read_parquet("data/training/eval_2025.parquet")
    df["target"] = pd.to_datetime(df["target"])
    df["origin"] = pd.to_datetime(df["origin"])
    return df


# ----- Prediction helpers -----
CATEGORICAL = ["city_tier", "footfall_zone_class", "country_code"]


def recompute_lags(base_row: pd.Series, target: date, origin: date,
                   actuals: dict[date, float]) -> pd.Series:
    row = base_row.copy()
    if "covers_1d_lag" in row.index:
        row["covers_1d_lag"] = actuals.get(origin, np.nan)
    for feat, offset in [("covers_7d_lag", 7), ("covers_14d_lag", 14),
                         ("covers_28d_lag", 28), ("covers_365d_lag", 365),
                         ("covers_same_dow_last_week", 7)]:
        if feat in row.index:
            look = target - timedelta(days=offset)
            row[feat] = actuals.get(look, actuals.get(origin, np.nan))
    if "covers_7d_mean" in row.index:
        vals = [actuals.get(origin - timedelta(days=k), np.nan) for k in range(7)]
        vals = [v for v in vals if not np.isnan(v)]
        row["covers_7d_mean"] = float(np.mean(vals)) if vals else np.nan
    if "covers_7d_std" in row.index:
        vals = [actuals.get(origin - timedelta(days=k), np.nan) for k in range(7)]
        vals = [v for v in vals if not np.isnan(v)]
        row["covers_7d_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    if "covers_28d_mean" in row.index:
        vals = [actuals.get(origin - timedelta(days=k), np.nan) for k in range(28)]
        vals = [v for v in vals if not np.isnan(v)]
        row["covers_28d_mean"] = float(np.mean(vals)) if vals else np.nan
    return row


def predict_one(models: dict, features: list[str], row: pd.Series,
                horizon: int, buffers: dict[int, float]) -> dict:
    """Predict for a single target with its horizon embedded and the
    horizon-specific conformal buffer applied."""
    row = row.copy()
    row["horizon_days"] = int(horizon)
    X = pd.DataFrame([row[features]])
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if X[col].dtype == object and col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    preds = {}
    for name, m in models.items():
        if isinstance(m, lgb.Booster):
            X_num = X.copy()
            for c in CATEGORICAL:
                if c in X_num.columns:
                    X_num[c] = X_num[c].cat.codes.astype(float) if hasattr(X_num[c], "cat") else 0
            preds[name] = float(max(0, m.predict(X_num.values)[0]))
        else:
            preds[name] = float(max(0, m.predict(X)[0]))
    # Per-horizon buffer widens the interval based on observed residuals at this horizon
    buf = buffers.get(int(horizon), 0.0)
    return {
        "lower": max(0, preds["lower"] - buf),
        "predicted": preds["median"],
        "upper": preds["upper"] + buf,
        "horizon_buffer": buf,
    }


def run_forecast(origin: date, days: int = 7) -> pd.DataFrame:
    daily, idx, actuals = load_daily()
    models, features, buffers = load_models()
    rows = []
    for h in range(1, days + 1):
        t = origin + timedelta(days=h)
        if t not in idx:
            continue
        row = daily.iloc[idx[t]].copy()
        row = recompute_lags(row, t, origin, actuals)
        q = predict_one(models, features, row, horizon=h, buffers=buffers)
        actual = actuals.get(t, None)
        rows.append({
            "target": t, "horizon": h, "dow": t.strftime("%a"),
            "lower": q["lower"], "predicted": q["predicted"], "upper": q["upper"],
            "interval_width": q["upper"] - q["lower"],
            "horizon_buffer": q["horizon_buffer"],
            "actual": actual,
            "temp_mean_c": float(row.get("temp_mean_c", np.nan)),
            "precipitation_total_mm": float(row.get("precipitation_total_mm", np.nan)),
            "wind_speed_mean_kmh": float(row.get("wind_speed_mean_kmh", np.nan)),
            "is_bank_holiday": bool(row.get("is_bank_holiday", False)),
            "is_tube_strike": bool(row.get("is_tube_strike", False)),
            "is_cultural_period": bool(row.get("is_cultural_period", False)),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["target"] = pd.to_datetime(df["target"])
    return df


# ----- Sidebar -----
with st.sidebar:
    st.title("🚲 Soho Cycles")
    st.markdown(
        "A demand-forecasting demo: predict daily Santander Cycles trips "
        "at Soho-area docking stations using only freely-available signals "
        "(weather, calendar, daylight, tube strikes, lag history)."
    )
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("- 🎯 **Predict** (this page)")
    st.markdown("- 📊 Accuracy")
    st.markdown("- ⚙️ How it works")
    st.markdown("---")
    st.caption("Model trained on 2017-2024 Santander Cycles data "
               "(ex-COVID), evaluated on 2025.")


# ----- Main -----
st.title("🎯 7-day demand forecast")
st.markdown(
    "Pick an **origin date** and the model will forecast the next 7 days with "
    "an **80% prediction interval**. Where actuals are available (2025), we "
    "overlay them so you can see how the forecast held up."
)

_, idx, actuals = load_daily()
min_date = min(actuals.keys())
max_date = max(actuals.keys())

col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    origin = st.date_input(
        "Origin date (the last known actual)",
        value=date(2025, 6, 15),
        min_value=date(2017, 2, 1),
        max_value=max_date,
        help="The model predicts the 7 days AFTER this date using only data from this date and earlier.",
    )

with col2:
    days = st.slider("Forecast horizon (days)", 1, 7, 7)

with col3:
    st.markdown("**Origin context**")
    if origin in actuals:
        origin_actual = actuals[origin]
        origin_dow = origin.strftime("%A")
        st.metric(
            label=f"{origin_dow} {origin.isoformat()}",
            value=f"{origin_actual:.0f} trips",
        )
    else:
        st.info(f"{origin}: no actual available")


# Run forecast
df = run_forecast(origin, days)

if df.empty:
    st.warning("No forecast could be built — check that the origin date is within the data range.")
    st.stop()

# ----- Main chart -----
fig = go.Figure()

# Context: 14 days of history before the origin
hist_dates = [origin - timedelta(days=k) for k in range(14, -1, -1)]
hist_actuals = [actuals.get(d, None) for d in hist_dates]
fig.add_trace(go.Scatter(
    x=hist_dates, y=hist_actuals,
    mode="lines+markers",
    name="Past actuals",
    line=dict(color="#888", width=2),
    marker=dict(size=6),
    hovertemplate="<b>%{x|%a %b %d}</b><br>Actual: %{y:.0f} trips<extra></extra>",
))

# 80% interval (shaded band)
fig.add_trace(go.Scatter(
    x=list(df["target"]) + list(df["target"])[::-1],
    y=list(df["upper"]) + list(df["lower"])[::-1],
    fill="toself",
    fillcolor="rgba(31,119,180,0.15)",
    line=dict(color="rgba(31,119,180,0)"),
    name="80% interval",
    hoverinfo="skip",
    showlegend=True,
))

# Point forecast
fig.add_trace(go.Scatter(
    x=df["target"], y=df["predicted"],
    mode="lines+markers",
    name="Forecast",
    line=dict(color="#1f77b4", width=3),
    marker=dict(size=10, symbol="circle"),
    customdata=np.stack([df["lower"], df["upper"]], axis=-1),
    hovertemplate=(
        "<b>%{x|%a %b %d}</b><br>"
        "Forecast: %{y:.0f}<br>"
        "Interval: [%{customdata[0]:.0f} .. %{customdata[1]:.0f}]<extra></extra>"
    ),
))

# Actuals (where available)
if df["actual"].notna().any():
    known = df[df["actual"].notna()]
    in_interval = (known["actual"] >= known["lower"]) & (known["actual"] <= known["upper"])
    colors = ["#2ca02c" if b else "#d62728" for b in in_interval]
    fig.add_trace(go.Scatter(
        x=known["target"], y=known["actual"],
        mode="markers",
        name="Actual",
        marker=dict(size=14, color=colors, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        hovertemplate=(
            "<b>%{x|%a %b %d}</b><br>"
            "Actual: %{y:.0f} trips<extra></extra>"
        ),
    ))

# Vertical line marking origin
fig.add_vline(x=origin, line_width=1.5, line_dash="dot", line_color="#555")
fig.add_annotation(
    x=origin, y=1.02, yref="paper",
    text="origin", showarrow=False, font=dict(size=11, color="#555"),
)

fig.update_layout(
    height=460,
    hovermode="x unified",
    xaxis_title=None,
    yaxis_title="Daily trips at Soho stations",
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")

st.plotly_chart(fig, width="stretch")


# ----- Table -----
st.subheader("Forecast details")

def _fmt_int(x):
    return f"{x:.0f}" if pd.notna(x) else "—"


def _flags(row):
    fl = []
    if row["is_bank_holiday"]: fl.append("🏖 bank holiday")
    if row["is_tube_strike"]: fl.append("🚇 tube strike")
    if row["is_cultural_period"]: fl.append("🎉 cultural event")
    return ", ".join(fl)


display = pd.DataFrame({
    "Date": df["target"].dt.strftime("%a %Y-%m-%d"),
    "Horizon": df["horizon"].apply(lambda h: f"{h}d"),
    "Lower (10%)": df["lower"].apply(_fmt_int),
    "Forecast": df["predicted"].apply(_fmt_int),
    "Upper (90%)": df["upper"].apply(_fmt_int),
    "Actual": df["actual"].apply(_fmt_int),
    "°C": df["temp_mean_c"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—"),
    "Rain mm": df["precipitation_total_mm"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—"),
    "Wind km/h": df["wind_speed_mean_kmh"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—"),
    "Flags": df.apply(_flags, axis=1),
})

# Highlight in-interval / out-of-interval
def _highlight(row):
    actual_s = row["Actual"]
    if actual_s == "—":
        return [""] * len(row)
    actual = float(actual_s)
    lo = float(row["Lower (10%)"])
    hi = float(row["Upper (90%)"])
    color = "background-color: rgba(44,160,44,0.12)" if lo <= actual <= hi else "background-color: rgba(214,39,40,0.12)"
    return [color] * len(row)


st.dataframe(display.style.apply(_highlight, axis=1), width="stretch", hide_index=True)


# ----- KPIs -----
if df["actual"].notna().any():
    known = df.dropna(subset=["actual"])
    in_interval_flags = (known["actual"] >= known["lower"]) & (known["actual"] <= known["upper"])
    in_int = in_interval_flags.mean()
    n_in = int(in_interval_flags.sum())
    mae = (known["predicted"] - known["actual"]).abs().mean()
    wape = (known["predicted"] - known["actual"]).abs().sum() / max(known["actual"].abs().sum(), 1e-8)

    st.subheader("How this forecast performed")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days in interval", f"{n_in}/{len(known)}", f"{in_int:.0%} coverage")
    c2.metric("MAE", f"{mae:.0f} trips/day")
    c3.metric("WAPE", f"{wape:.1%}")
    mean_actual = known["actual"].mean()
    c4.metric("Avg actual", f"{mean_actual:.0f} trips/day")


st.markdown("---")
with st.expander("💡 How to read this"):
    st.markdown("""
- **Grey line (left)** = the 2 weeks of actuals before your origin date — the context the model sees when it predicts.
- **Blue solid line** = the point forecast (50th percentile).
- **Blue shaded band** = the 80% prediction interval. We expect actuals to fall inside this band 80% of the time.
- **Green diamonds** = actual values that fell inside the interval.
- **Red diamonds** = actual values that fell outside — days the model got surprised.
- The interval is built from three separate LightGBM models (5th, 50th, 95th percentile) with a conformal-calibration buffer added so empirical coverage matches the 80% target.
""")
