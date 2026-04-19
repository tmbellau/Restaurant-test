"""Soho Cycling Demand — single-page demo.

One page, six sections:
1. Header + KPIs
2. Live forecast demo (date picker → chart)
3. Full-year performance (horizon + monthly)
4. What goes into the model (features + importance)
5. How it works
6. Limitations
"""

from __future__ import annotations

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================================
# Config + loaders
# ==========================================================================
st.set_page_config(
    page_title="Soho Cycles Forecast",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def load_daily() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_parquet("data/training/daily_training_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["covers_7d_lag"]).reset_index(drop=True)
    idx = {d.date(): i for d, i in zip(df["date"], df.index)}
    actuals = {d.date(): float(c) for d, c in zip(df["date"], df["cover_count"])}
    return df, idx, actuals


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[dict, list[str], dict[int, float], dict]:
    model_dir = Path("data/models")
    features = json.loads((model_dir / "horizon_aware_feature_names.json").read_text())
    metrics = json.loads((model_dir / "horizon_aware_metrics.json").read_text())
    buffers = {int(k): float(v) for k, v in metrics["per_horizon_buffer"].items()}
    models = {}
    for name, alpha in [("lower", 5), ("median", 50), ("upper", 95)]:
        txt_path = model_dir / f"lgbm_horizon_q{alpha:02d}.txt"
        models[name] = lgb.Booster(model_file=str(txt_path))
    return models, features, buffers, metrics


@st.cache_data(show_spinner=False)
def load_eval_2025() -> pd.DataFrame:
    df = pd.read_parquet("data/training/eval_2025_horizon_aware.parquet")
    df["target"] = pd.to_datetime(df["target"])
    return df


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
    row = row.copy()
    row["horizon_days"] = int(horizon)
    X = pd.DataFrame([row[features]])
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes.astype(float)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    preds = {name: float(max(0, m.predict(X.values)[0])) for name, m in models.items()}
    buf = buffers.get(int(horizon), 0.0)
    return {
        "lower": max(0, preds["lower"] - buf),
        "predicted": preds["median"],
        "upper": preds["upper"] + buf,
    }


def run_forecast(origin: date, days: int = 7) -> pd.DataFrame:
    daily, idx, actuals = load_daily()
    models, features, buffers, _ = load_models()
    rows = []
    for h in range(1, days + 1):
        t = origin + timedelta(days=h)
        if t not in idx:
            continue
        row = daily.iloc[idx[t]].copy()
        row = recompute_lags(row, t, origin, actuals)
        q = predict_one(models, features, row, horizon=h, buffers=buffers)
        rows.append({
            "target": t, "horizon": h, "dow": t.strftime("%a"),
            "lower": q["lower"], "predicted": q["predicted"], "upper": q["upper"],
            "actual": actuals.get(t, None),
            "temp_mean_c": float(row.get("temp_mean_c", np.nan)),
            "precipitation_total_mm": float(row.get("precipitation_total_mm", np.nan)),
            "is_bank_holiday": bool(row.get("is_bank_holiday", False)),
            "is_tube_strike": bool(row.get("is_tube_strike", False)),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["target"] = pd.to_datetime(df["target"])
    return df


# ==========================================================================
# SECTION 1 — Header & top-line KPIs
# ==========================================================================
st.title("🚲 Soho Cycles Demand Forecast")
st.markdown(
    "Predicts **daily Santander Cycles trips** at Soho docking stations using "
    "only freely-available inputs: weather, calendar, daylight, and tube strikes. "
    "Trained on **2017-2024** (COVID years excluded). "
    "Every number below is evaluated on **2025 — an entire year the model never saw during training.**"
)

eval_df = load_eval_2025()
eval_ok = eval_df[~eval_df["is_outage"]]
h1 = eval_ok[eval_ok["horizon"] == 1]

overall_cov = eval_ok["in_interval"].mean()
overall_wape = eval_ok["abs_err"].sum() / eval_ok["actual"].abs().sum()
h1_wape = (h1["predicted"] - h1["actual"]).abs().sum() / h1["actual"].abs().sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("1-day-ahead WAPE", f"{h1_wape:.1%}",
          help="Weighted average % error predicting tomorrow. Computed on 1-day-ahead predictions only (the most accurate horizon).")
c2.metric("80% interval coverage", f"{overall_cov:.0%}",
          help="Actuals fall inside the prediction interval this often — target is 80%, achieved via conformal calibration.")
c3.metric("Training data", "6 years (2017-2024)",
          help="Excluding 2020-2021 COVID. 2,264 training days of real Santander trip data.")
c4.metric("Holdout (unseen)", "All of 2025",
          help="Every metric on this page comes from 2025 data the model never saw during training.")

st.divider()


# ==========================================================================
# SECTION 2 — Live demo
# ==========================================================================
st.header("Try it: pick any week in 2025")
st.markdown(
    "Choose an **origin date** and see the model forecast the next 7 days, "
    "with an 80% prediction interval. Actuals are overlaid so you can see "
    "exactly how the forecast held up. On average across the year, intervals "
    "are modestly wider at longer horizons (see next section) — but per-day "
    "width varies with the specific feature combination, so you'll often see "
    "non-monotonic widths within a single week."
)

daily, idx, actuals = load_daily()
min_date = min(actuals.keys())
max_date = max(actuals.keys())

# Good default — a week where weather was stable and the model did well
DEFAULT_ORIGIN = date(2025, 10, 5)

col1, col2 = st.columns([2, 5])
with col1:
    origin = st.date_input(
        "Origin date",
        value=DEFAULT_ORIGIN,
        min_value=date(2017, 2, 1),
        max_value=max_date,
        help="The model forecasts the 7 days AFTER this date.",
    )
    if origin in actuals:
        st.metric(
            label=f"{origin.strftime('%A')} {origin.isoformat()}",
            value=f"{actuals[origin]:.0f} trips (actual)",
        )
    st.caption(
        "🎯 Try different dates:\n"
        "- Oct 5 / Apr 20 / Jun 22: typical good weeks\n"
        "- Sep 7: during the week-long tube strike\n"
        "- Jan 12: cold/rainy winter week\n"
    )

df = run_forecast(origin, 7)

with col2:
    if df.empty:
        st.warning("No forecast available for that date.")
    else:
        fig = go.Figure()

        # Past context (14 days of actuals)
        hist_dates = [origin - timedelta(days=k) for k in range(14, -1, -1)]
        hist_vals = [actuals.get(d) for d in hist_dates]
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_vals, mode="lines+markers",
            name="Past actuals",
            line=dict(color="#888", width=2), marker=dict(size=5),
            hovertemplate="<b>%{x|%a %b %d}</b><br>Actual: %{y:.0f}<extra></extra>",
        ))

        # 80% interval
        fig.add_trace(go.Scatter(
            x=list(df["target"]) + list(df["target"])[::-1],
            y=list(df["upper"]) + list(df["lower"])[::-1],
            fill="toself", fillcolor="rgba(31,119,180,0.18)",
            line=dict(color="rgba(31,119,180,0)"),
            name="80% interval", hoverinfo="skip",
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=df["target"], y=df["predicted"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#1f77b4", width=3), marker=dict(size=10),
            customdata=list(zip(df["lower"].round(0), df["upper"].round(0), df["horizon"])),
            hovertemplate=(
                "<b>%{x|%a %b %d}</b> (%{customdata[2]}d ahead)<br>"
                "Forecast: %{y:.0f}<br>"
                "Interval: [%{customdata[0]:.0f} .. %{customdata[1]:.0f}]<extra></extra>"
            ),
        ))

        # Actuals overlay
        if df["actual"].notna().any():
            kn = df[df["actual"].notna()]
            in_iv = (kn["actual"] >= kn["lower"]) & (kn["actual"] <= kn["upper"])
            colors = ["#2ca02c" if b else "#d62728" for b in in_iv]
            fig.add_trace(go.Scatter(
                x=kn["target"], y=kn["actual"],
                mode="markers", name="Actual",
                marker=dict(size=13, color=colors, symbol="diamond",
                            line=dict(color="white", width=1.5)),
                hovertemplate="<b>%{x|%a %b %d}</b><br>Actual: %{y:.0f}<extra></extra>",
            ))

        # Origin marker
        fig.add_vline(x=origin, line_width=1.5, line_dash="dot", line_color="#555")

        fig.update_layout(
            height=400, hovermode="x unified",
            xaxis_title=None, yaxis_title="Daily trips at Soho stations",
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        st.plotly_chart(fig, width="stretch")

# Forecast table + summary underneath
if not df.empty:
    if df["actual"].notna().any():
        kn = df.dropna(subset=["actual"])
        in_iv = (kn["actual"] >= kn["lower"]) & (kn["actual"] <= kn["upper"])
        mae = (kn["predicted"] - kn["actual"]).abs().mean()
        wape = (kn["predicted"] - kn["actual"]).abs().sum() / max(kn["actual"].abs().sum(), 1e-8)
        c1, c2, c3 = st.columns(3)
        c1.metric("This week's coverage", f"{int(in_iv.sum())}/{len(kn)} days inside interval")
        c2.metric("This week's MAE", f"{mae:.0f} trips/day")
        c3.metric("This week's WAPE", f"{wape:.1%}")

    # ----- AI Explanation Button -----
    st.markdown("#### Understand this forecast")
    if st.button("🤖 Explain why the model predicted this", type="primary"):
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
        if not api_key:
            api_key = st.session_state.get("_anthropic_key", "")
        if not api_key:
            st.warning("No API key found. Add ANTHROPIC_API_KEY to Streamlit secrets or enter below.")
            api_key = st.text_input("Anthropic API key", type="password", key="_key_input")
            if api_key:
                st.session_state["_anthropic_key"] = api_key

        if api_key:
            import anthropic

            # Build context for Claude
            week_data = []
            for _, r in df.iterrows():
                day_info = {
                    "date": r["target"].strftime("%A %Y-%m-%d"),
                    "horizon_days_ahead": int(r["horizon"]),
                    "predicted_trips": round(float(r["predicted"])),
                    "lower_80pct": round(float(r["lower"])),
                    "upper_80pct": round(float(r["upper"])),
                    "actual_trips": round(float(r["actual"])) if pd.notna(r["actual"]) else "unknown",
                    "temperature_c": round(float(r["temp_mean_c"]), 1) if pd.notna(r["temp_mean_c"]) else None,
                    "rainfall_mm": round(float(r["precipitation_total_mm"]), 1) if pd.notna(r["precipitation_total_mm"]) else None,
                    "bank_holiday": bool(r["is_bank_holiday"]),
                    "tube_strike": bool(r["is_tube_strike"]),
                }
                week_data.append(day_info)

            origin_actual = actuals.get(origin, 0)
            prompt = f"""You are analysing predictions from a cycling demand forecasting model.
The model predicts daily Santander Cycles trip counts at Soho docking stations in London.

The forecast origin is {origin.strftime('%A %Y-%m-%d')} with {origin_actual:.0f} actual trips that day.

Here are the predictions for the next {len(week_data)} days:

{json.dumps(week_data, indent=2)}

The model uses these key inputs: weather (temperature, rain, wind), UK bank holidays,
school holidays, London cultural events, daylight hours, tube strike flags, and
historical demand lags (1d, 7d, 14d, 28d, 365d rolling averages).

The top features by importance are: temperature change (day-over-day), wind speed,
minimum temperature, temperature anomaly vs 30-day average, 28-day rolling mean,
holiday proximity, and precipitation.

Write a clear 3-4 paragraph explanation for a non-technical reader covering:
1. What the model predicted and how it compares to what actually happened
2. What likely drove the prediction levels (which inputs mattered most this week)
3. Where it got things right and where it missed, with specific reasons
4. Any notable patterns (weekend dips, weather effects, holiday effects)

Be specific about the numbers. Use plain language. Don't hedge excessively."""

            with st.spinner("Generating explanation..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=800,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    explanation = response.content[0].text
                    st.markdown(explanation)
                except Exception as e:
                    st.error(f"API error: {e}")

st.divider()


# ==========================================================================
# SECTION 3 — Full-year performance
# ==========================================================================
st.header("How it did across 2025")
st.markdown(
    "Every day in 2025 was forecast 7 times (1 to 7 days ahead) and compared "
    "to the actual — 2,548 forecasts in total. Here's the full picture."
)

c1, c2 = st.columns(2)

# Left: WAPE by horizon
h_stats = eval_ok.groupby("horizon").agg(
    coverage=("in_interval", "mean"),
    wape=("abs_err", lambda x: x.sum() / eval_ok.loc[x.index, "actual"].abs().sum()),
    mae=("abs_err", "mean"),
).reset_index()

with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=h_stats["horizon"].astype(str) + "d ahead",
        y=(h_stats["wape"] * 100).round(2),
        marker_color="#1f77b4",
        text=(h_stats["wape"] * 100).round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig.update_layout(
        title="Accuracy by forecast horizon",
        yaxis_title="WAPE (%)", xaxis_title=None,
        height=320, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_yaxes(range=[0, max(h_stats["wape"] * 100) * 1.25])
    st.plotly_chart(fig, width="stretch")
    st.caption("WAPE (lower = better) grows gently with horizon, from ~12% at 1 day to ~13% at 7 days.")

with c2:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=h_stats["horizon"].astype(str) + "d ahead",
        y=(h_stats["coverage"] * 100).round(1),
        marker_color=["#2ca02c" if c >= 0.77 else "#ff7f0e" for c in h_stats["coverage"]],
        text=(h_stats["coverage"] * 100).round(0).astype(int).astype(str) + "%",
        textposition="outside",
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="#555",
                  annotation_text="80% target", annotation_position="top right")
    fig.update_layout(
        title="Prediction interval coverage",
        yaxis_title="Coverage (%)", xaxis_title=None,
        height=320, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, width="stretch")
    st.caption("Actuals fall inside the interval ~80% of the time at every horizon — by design (conformal calibration).")

# Monthly WAPE breakdown
st.markdown("##### Accuracy month by month (1-day-ahead)")
monthly = h1.copy()
monthly["month"] = monthly["target"].dt.month
monthly["month_name"] = monthly["target"].dt.month_name().str[:3]
m_stats = monthly.groupby(["month", "month_name"]).agg(
    wape=("abs_err", lambda x: x.sum() / monthly.loc[x.index, "actual"].abs().sum()),
).reset_index().sort_values("month")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=m_stats["month_name"],
    y=(m_stats["wape"] * 100).round(1),
    marker_color=[
        "#d62728" if w > 0.20 else "#ff7f0e" if w > 0.14 else "#2ca02c"
        for w in m_stats["wape"]
    ],
    text=(m_stats["wape"] * 100).round(1).astype(str) + "%",
    textposition="outside",
))
fig.update_layout(
    height=260, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title="WAPE (%)", xaxis_title=None,
)
fig.update_yaxes(range=[0, max(m_stats["wape"] * 100) * 1.3])
st.plotly_chart(fig, width="stretch")
st.markdown(
    """
**Why is winter (especially January) so much less accurate?**

Three compounding factors:

1. **Lower volume amplifies percentage error.** January averages ~680 trips/day vs ~1,800 in June.
   A 100-trip miss is 15% error in January but only 6% in June — same absolute error, worse WAPE.

2. **Weather is more volatile.** Rain, wind, and cold suppress cycling demand sharply but
   unpredictably. One unexpectedly mild January day can produce 2× the trips of the day before.
   Summer weather is more stable.

3. **Fewer training examples of extreme cold.** The model has 6 years of data but London rarely
   gets sustained sub-zero weather, so the training set doesn't cover enough extreme-cold
   scenarios for the model to learn precise responses.

All WAPE numbers on this page use the same computation: `sum(|predicted - actual|) / sum(actual)`,
on **1-day-ahead** predictions only (the most accurate horizon).
"""
)

st.divider()


# ==========================================================================
# SECTION 4 — What goes into the model
# ==========================================================================
st.header("What goes into the model")
st.markdown(
    "49 features per day, grouped into five themes. All free, all automatically collected, "
    "no manual inputs."
)

col_feats, col_imp = st.columns([2, 3])

with col_feats:
    st.markdown("##### Input features")
    groups = {
        "🌦 **Weather** (15)": "Temperature (mean / min / max / anomaly), rainfall total, "
                                "wind speed, weather code, rain streak (consecutive wet days), "
                                "temperature shock (day-over-day swing), severe weather flag.",
        "📅 **Calendar** (17)": "Day of week, month, week, day of year (with cyclical encodings), "
                                 "bank holidays + days-to/from, England school holidays, "
                                 "cultural events (Marathon, Pride, Carnival, Wimbledon, NYE), "
                                 "known tube-strike days + days-since.",
        "☀️ **Daylight** (3)": "Sunrise hour, sunset hour, daylight hours — astronomically "
                                "calculated from the date. Deterministic, never wrong.",
        "🔁 **Demand history** (9)": "Cover count 1 / 7 / 14 / 28 / 365 days ago, rolling 7d and "
                                      "28d means, rolling 7d volatility. Horizon-aware: stale at "
                                      "longer horizons.",
        "📊 **Horizon** (1)": "How far ahead we're forecasting (1 to 7 days). The model learns "
                               "that stale lag features come with wider uncertainty.",
    }
    for title, desc in groups.items():
        st.markdown(f"{title}")
        st.caption(desc)

with col_imp:
    st.markdown("##### What the model actually uses most")
    _, _, _, metrics = load_models()
    booster = lgb.Booster(model_file="data/models/lgbm_horizon_q50.txt")
    imp_df = pd.DataFrame({
        "feature": booster.feature_name(),
        "importance": booster.feature_importance(importance_type="split"),
    }).sort_values("importance", ascending=True).tail(15)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=imp_df["feature"], x=imp_df["importance"],
        orientation="h",
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        height=500, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Relative importance (tree splits)", yaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Weather features (wind, temperature, temperature change) dominate — which is right: "
        "cycling demand really does depend on the weather. Holiday distance and day-of-year seasonality "
        "also carry weight. Lag features anchor the model in recent context."
    )

st.divider()


# ==========================================================================
# SECTION 5 — How it works
# ==========================================================================
st.header("How it works")
st.markdown(
    """
**The target.** Daily count of Santander Cycles trips at Soho-area docking stations
(Moor Street, Wardour Street, Broadwick Street, Golden Square and nearby West End stations).
Data from TfL's public open-data bucket, 4.25 million trips since 2017.

**The model.** Three LightGBM gradient-boosted trees predicting the 5th, 50th, and 95th
percentile of demand given the 49 input features. Each is an ensemble of ~500 trees;
training minimises pinball loss (asymmetric, targets the specified percentile).

**The prediction intervals.** The raw quantile predictions are widened by a per-horizon
conformal buffer computed from a fully held-out year (2024). For each horizon, we take
the exact residual quantile needed to cover 80% of actuals — no hand-tuned multiplier.
On average this produces larger buffers at longer horizons (±99 at 1 day ahead growing
to ±125 at 7 days ahead), which is why average interval widths grow 629 → 698 trips
across horizons on the 2025 holdout.

*Honest caveat*: the raw quantile models themselves don't reliably produce wider intervals
at longer horizons — they react to feature combinations, not horizon. So the widening is
real **on average** but on any given week the width is dominated by per-day feature
variation. This is the best honest calibration I could produce without hand-tuning.
"""
)

st.divider()


# ==========================================================================
# SECTION 6 — Limitations
# ==========================================================================
st.header("Honest limitations")
st.markdown(
    """
- **Winter months are harder.** January WAPE is ~21% vs ~8% in May. Cold/wet weather makes
  cycling demand volatile and the training data doesn't contain enough extreme-weather days.
- **Rare disruptions hurt the model.** The September 2025 week-long tube strike was a type of
  event seen only twice in training — the model under-predicted the surge. More years of data
  would help.
- **Structural shifts can't be predicted from historical patterns.** 2025 Soho cycling was
  ~15% higher than 2024 overall (likely TfL e-bike expansion). The model adapted within a
  month or two as the lag features caught up, but the first days of a regime change will
  always be wrong.
- **This is cycling, not restaurant demand.** The architecture transfers cleanly to any
  daily-demand target with similar drivers, but the current numbers are strictly about
  cycling trips at Soho docking stations.
- **Data publication lag.** TfL publishes Santander CSVs ~1-2 months after the fact, so
  live "next-day" predictions in production would be limited by data freshness, not model
  accuracy.
- **Uncertainty widening is coarse.** The intervals are wider at longer horizons **on
  average across the year** because of a per-horizon calibration buffer — but for any
  single forecast, the width is feature-dependent and may not grow monotonically with
  horizon. The model itself doesn't produce "I'm less confident about next Tuesday than
  about tomorrow" out of the box; that's added via calibration. A proper horizon-aware
  model would train quantiles per-horizon directly.
"""
)

st.caption(
    "Code: github.com/tmbellau/restaurant-test • "
    "Built with LightGBM, Plotly, Streamlit. "
    "All data sources are free and public."
)
