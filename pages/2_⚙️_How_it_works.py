"""How the model works — features, training, and interpretation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="How it works — Soho Cycles Forecast",
                   page_icon="⚙️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_median_model():
    """Prefer horizon-aware median model; fall back to earlier models."""
    for txt in ["data/models/lgbm_horizon_q50.txt", "data/models/lgbm_daily_q50.txt"]:
        p = Path(txt)
        if p.exists():
            booster = lgb.Booster(model_file=str(p))
            return {
                "booster": booster,
                "importances": booster.feature_importance(importance_type="split"),
                "feature_names": booster.feature_name(),
            }
    # Ultimate fallback
    with open("data/models/lgbm_daily_q50.pkl", "rb") as fh:
        model = pickle.load(fh)
    return {
        "booster": None,
        "importances": model.feature_importances_,
        "feature_names": json.loads(Path("data/models/quantile_feature_names.json").read_text()),
    }


@st.cache_data(show_spinner=False)
def load_metrics():
    for p in ["data/models/horizon_aware_metrics.json", "data/models/quantile_metrics.json"]:
        if Path(p).exists():
            return json.loads(Path(p).read_text())
    return {}


@st.cache_data(show_spinner=False)
def load_daily_summary():
    df = pd.read_parquet("data/training/daily_training_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


with st.sidebar:
    st.title("🚲 Soho Cycles")
    st.markdown("- 🎯 Predict")
    st.markdown("- 📊 Accuracy")
    st.markdown("- ⚙️ **How it works** (this page)")


st.title("⚙️ How the model works")
st.markdown(
    "A plain-language walkthrough of what goes in, what comes out, and why."
)

# ============ High-level ============
st.header("1. What the model predicts")
st.markdown(
    """
The **target** is the daily count of Santander Cycles trips (starts + ends combined)
at Soho-area docking stations: Moor Street, Wardour Street, Broadwick Street,
Golden Square, and nearby West End stations.

Trip records come from TfL's public open-data bucket and go back to 2015.
We only use **post-2017** data because pre-2017 station-ID formats differ.
**2020-2021 is excluded** (COVID lockdowns distort the signal).
"""
)

daily_df = load_daily_summary()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Training period", "2017-01 to 2024-12", help="Excluding 2020 and 2021 (COVID)")
col2.metric("Holdout year", "2025", help="No day in 2025 was seen during training")
col3.metric("Total training days", f"{len(daily_df[daily_df['date'] < '2025-01-01']):,}")
col4.metric("Real trips in dataset",
            f"{int(daily_df['cover_count'].sum()):,}")

st.markdown("---")


# ============ Features ============
st.header("2. What goes into the model (features)")
st.markdown(
    "For each day, the model sees ~48 numbers describing that day. These "
    "break into five groups:"
)

feature_groups = {
    "🌦 Weather (15)": [
        ("temp_mean_c", "Mean temperature °C (from Open-Meteo archive)"),
        ("temp_min_c", "Min temperature °C"),
        ("temp_max_c", "Max temperature °C"),
        ("temp_anomaly_mean_c", "Deviation from 30-day trailing temperature average"),
        ("apparent_temp_mean_c", "Mean 'feels-like' temperature"),
        ("wind_speed_mean_kmh", "Mean wind speed km/h"),
        ("precipitation_total_mm", "Daily total rainfall (mm)"),
        ("weather_code_max", "Open-Meteo weather code (peak severity)"),
        ("is_wet_day", "Binary: rainfall > 2mm"),
        ("rain_streak", "Consecutive preceding wet days (0-7)"),
        ("temp_change_c", "Day-over-day change in mean temperature"),
        ("temp_shock", "Binary: |temp_change| > 5°C"),
        ("severe_weather", "Binary: storm / snow / heavy rain"),
        ("cold_weekend", "Binary: weekend AND cold"),
        ("hot_day", "Binary: max temp > 22°C"),
    ],
    "📅 Calendar (17)": [
        ("dow", "Day of week (0=Mon)"),
        ("is_weekend", "Binary"),
        ("month, week_of_year, day_of_year", "Calendar position"),
        ("dow_sin, dow_cos", "Cyclical encoding of weekday"),
        ("doy_sin, doy_cos", "Cyclical encoding of day-of-year"),
        ("is_bank_holiday, is_school_holiday", "From gov.uk bank-holidays.json + England school term dates"),
        ("days_to_next_holiday, days_from_last_holiday", "Proximity to the nearest bank holiday"),
        ("is_cultural_period", "London Marathon, Notting Hill Carnival, Pride, Wimbledon, etc."),
        ("days_to_next_cultural, days_from_last_cultural", "Proximity to cultural events"),
        ("is_tube_strike", "Known London Underground strike day (Wikipedia-verified)"),
        ("days_since_tube_strike", "Days since last strike (saturated at 30)"),
    ],
    "☀️ Daylight (3)": [
        ("daylight_hours", "Sunset − sunrise in hours (astronomical, deterministic)"),
        ("sunrise_hour, sunset_hour", "Local London time of sunrise / sunset"),
    ],
    "🔁 Demand history / lags (9)": [
        ("covers_1d_lag", "Yesterday's trips (origin-anchored)"),
        ("covers_7d_lag", "Same day last week"),
        ("covers_14d_lag", "Same day 2 weeks ago"),
        ("covers_28d_lag", "Same day 4 weeks ago"),
        ("covers_365d_lag", "Same day last year"),
        ("covers_7d_mean, covers_28d_mean", "Rolling average over preceding 7 and 28 days"),
        ("covers_7d_std", "Rolling 7-day standard deviation (volatility)"),
        ("covers_same_dow_last_week", "Same day-of-week, last week"),
    ],
    "📍 Location (static)": [
        ("city_tier, footfall_zone_class, country_code", "Descriptive tags (constant for Soho)"),
    ],
}

cols = st.columns(2)
for i, (group, items) in enumerate(feature_groups.items()):
    with cols[i % 2]:
        st.subheader(group)
        for name, desc in items:
            st.markdown(f"- **`{name}`** — {desc}")

st.markdown("---")


# ============ Model architecture ============
st.header("3. What the model is")
st.markdown(
    """
**LightGBM gradient-boosted decision trees.**

We train **three** models with different objectives:
- `lgbm_daily_q05.pkl` — predicts the **5th percentile** of daily demand
- `lgbm_daily_q50.pkl` — predicts the **median** (point forecast)
- `lgbm_daily_q95.pkl` — predicts the **95th percentile**

Each model is an ensemble of ~500 decision trees. During training, each tree
learns a simple "if-this-then-that" rule on a subset of features, and the
trees' outputs are summed together. The model uses the
**quantile loss** (asymmetric pinball loss) to target the specified percentile
rather than the mean.

To turn the three quantile predictions into a calibrated 80% interval, we add
a **per-horizon conformal buffer**: for each forecast horizon (1 to 7 days),
we look at how large the model's errors were on a fully held-out year (2024
— not seen during training) and widen the interval by the amount needed to
achieve 80% coverage at that specific horizon.

**Why this gives organic horizon-dependent uncertainty:**

Models are trained on multi-horizon data. For each historical day, we
generate 7 training rows — one per forecast horizon — with the lag features
substituted as they would appear at that horizon (at h=7, covers_1d_lag
uses the value from 7 days ago, not 1 day ago). The model therefore sees
how feature staleness affects error magnitude, and the raw quantile
predictions reflect this.

The per-horizon buffer is then computed directly from the distribution
of errors at each horizon on the held-out year. Since 7-day-ahead errors
are genuinely larger than 1-day-ahead errors (the data shows this), the
7-day buffer is genuinely larger — nothing hand-tuned, nothing arbitrary.
"""
)

metrics = load_metrics()
col1, col2, col3 = st.columns(3)
col1.metric("Features", metrics.get("n_features", "?"))
col2.metric("Training rows", metrics.get("n_training_rows", "?"))
col3.metric("Calibration year", metrics.get("calibrate_year", "—"),
            help="Held-out year used to compute the per-horizon conformal buffer.")

# Show per-horizon buffers — these are the organic uncertainty widenings
per_h = metrics.get("per_horizon_buffer", {})
if per_h:
    st.subheader("Per-horizon conformal buffer — learned from data")
    st.caption("These numbers are NOT hand-tuned. They come directly from how large the "
               "model's errors were at each horizon on a held-out year.")
    buf_df = pd.DataFrame([
        {"Horizon": f"{int(k)} day{'s' if int(k)>1 else ''} ahead",
         "Buffer (±trips)": f"{float(v):.0f}",
         "Avg interval width": f"{float(metrics.get('interval_width_per_horizon', {}).get(str(k), 0)):.0f}"}
        for k, v in sorted(per_h.items(), key=lambda x: int(x[0]))
    ])
    st.dataframe(buf_df, hide_index=True)

st.markdown("---")


# ============ Feature importance ============
st.header("4. What the model actually leans on")
st.markdown(
    "After training, LightGBM tells us how often each feature was used to split "
    "a decision tree. Higher = more important."
)

model_info = load_median_model()
imp = pd.DataFrame({
    "feature": model_info["feature_names"],
    "importance": model_info["importances"],
}).sort_values("importance", ascending=True).tail(20)

fig = px.bar(
    imp, x="importance", y="feature",
    orientation="h", title="Top 20 features (median model)",
    color="importance", color_continuous_scale="Blues",
)
fig.update_layout(
    height=640, showlegend=False,
    margin=dict(l=0, r=0, t=40, b=0),
    coloraxis_showscale=False,
    yaxis_title=None, xaxis_title="Split count (relative importance)",
)
st.plotly_chart(fig, width="stretch")

st.markdown("""
**Observations:**
- Weather features (wind, temperature, temperature change) are consistently in the top 10.
- Lag features (14d, 28d, 365d) anchor the model in recent history.
- Holiday-proximity features (`days_to_next_holiday`, `days_from_last_holiday`) appear, while the binary `is_bank_holiday` flag itself barely registers — the continuous proximity signal carries more information.
- The `is_tube_strike` flag has near-zero importance because we have few strike days in training (~25).
""")

st.markdown("---")


# ============ Data sources ============
st.header("5. Data sources (all free)")

sources = [
    ("TfL Santander Cycles", "https://cycling.data.tfl.gov.uk/",
     "Per-trip records, ~4.25M trips since 2017 at Soho-area stations. Weekly CSVs, OGLv2 licence."),
    ("Open-Meteo Archive", "https://open-meteo.com/",
     "Historical hourly weather for Soho (51.5131°N, −0.1318°E). 80+ years. No key, no cost."),
    ("Open-Meteo Forecast", "https://open-meteo.com/",
     "14-day weather forecast — used at inference time for future-dated predictions."),
    ("UK bank holidays", "https://www.gov.uk/bank-holidays.json",
     "Official England and Wales bank holidays."),
    ("England school holidays", "—",
     "Curated window list for the Soho catchment area."),
    ("London cultural calendar", "—",
     "Hand-curated: Marathon, Pride, Carnival, Wimbledon, NYE, Christmas markets."),
    ("Tube strike list", "https://en.wikipedia.org/wiki/London_Underground_strikes",
     "Verified network-wide LU strike days 2017-2025."),
    ("Astral (Python)", "https://pypi.org/project/astral/",
     "Sunrise/sunset/daylight calculations — deterministic from date + location."),
]

for name, url, desc in sources:
    if url != "—":
        st.markdown(f"- **[{name}]({url})** — {desc}")
    else:
        st.markdown(f"- **{name}** — {desc}")


st.markdown("---")


# ============ Training pipeline ============
st.header("6. Training pipeline")

st.markdown(
    """
A one-line command rebuilds everything from scratch:

```bash
bash scripts/refresh_and_predict.sh
```

Under the hood:
1. **`src.data.build_training_set`** — pulls every Santander CSV in the date
   range from TfL's S3 bucket, filters to Soho-area stations (by ID or name
   — pre-2022 CSVs use a different ID scheme), aggregates trips to hourly
   counts, fetches Open-Meteo weather, attaches holiday/cultural/temporal
   features.
2. **`src.data.build_daily_training_set --exclude-covid`** — rolls up hourly
   features to daily, adds daylight / rain-streak / temperature-shock / tube-
   strike features, drops the COVID window.
3. **`src.models.train_quantile_ensemble`** — trains the three LightGBM
   quantile models and computes the conformal buffer from walk-forward CV.
4. **`scripts/predict_future.py`** — generates forward-looking predictions
   from the latest-known actual.
"""
)
