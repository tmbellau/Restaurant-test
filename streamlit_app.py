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
    """Load the three quantile models (5th/50th/95th percentile) and the
    per-horizon calibration buffer."""
    model_dir = Path("data/models")
    features = json.loads((model_dir / "quantile_feature_names.json").read_text())
    metrics = json.loads((model_dir / "horizon_aware_metrics.json").read_text())
    buffers = {int(k): float(v) for k, v in metrics["per_horizon_buffer"].items()}
    models = {}
    for name, alpha in [("lower", 5), ("median", 50), ("upper", 95)]:
        txt_path = model_dir / f"lgbm_daily_q{alpha:02d}.txt"
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

c1, c2, c3 = st.columns(3)
c1.metric("1-day-ahead WAPE", f"{h1_wape:.1%}",
          help="Average % error when forecasting tomorrow from today.")
c2.metric("Training data", "6 years (2017-2024)",
          help="Excludes 2020-2021 COVID. 2,264 training days.")
c3.metric("Holdout (unseen)", "All of 2025",
          help="Every number on this page comes from 2025 data the model didn't see during training.")

# Executive summary — computed from the data
_within_20 = (h1["pct_error"].abs() <= 20).mean() * 100
_within_10 = (h1["pct_error"].abs() <= 10).mean() * 100
st.markdown(
    f"<div style='padding: 14px 18px; margin-top: 10px; border-radius: 12px; "
    f"background: linear-gradient(135deg, rgba(102,126,234,0.06), rgba(237,100,166,0.04)); "
    f"border-left: 3px solid #9f7aea; font-size: 0.95rem;'>"
    f"<strong>In short:</strong> the model predicts <strong>{_within_10:.0f}%</strong> of 2025 days "
    f"within ±10% of the actual and <strong>{_within_20:.0f}%</strong> within ±20%. "
    f"Strongest May-Oct (WAPE 6-10%), hardest Jan-Feb (14-21%) where low volume amplifies % error."
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()


# ==========================================================================
# SECTION 2 — Live demo
# ==========================================================================
st.header("Try it: pick a date in 2025")
st.markdown(
    "Choose a **starting date** (Day 0). The model then forecasts **all 7 days ahead at once** — "
    "tomorrow (Day 1) through to a week from now (Day 7) — using only what it knows up to Day 0. "
    "The further out the forecast, the less the model knows, so accuracy naturally drops."
)

daily, idx, actuals = load_daily()
min_date = min(actuals.keys())
max_date = max(actuals.keys())

# Good default — a week where weather was stable and the model did well
DEFAULT_ORIGIN = date(2025, 10, 5)

# Initialise the widget's state once (this is the same key the date_input uses).
if "origin_input" not in st.session_state:
    st.session_state["origin_input"] = DEFAULT_ORIGIN

# Clamp helper so arrows don't walk off the end
def _clamp_origin(d: date) -> date:
    lo = date(2017, 2, 1)
    return max(lo, min(d, max_date))

col_dprev, col_prev, col_date, col_next, col_dnext, col_ctx = st.columns([1, 1, 3, 1, 1, 4])
with col_dprev:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Day", key="prev_day"):
        st.session_state["origin_input"] = _clamp_origin(
            st.session_state["origin_input"] - timedelta(days=1)
        )
        st.rerun()
with col_prev:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Week", key="prev_wk"):
        st.session_state["origin_input"] = _clamp_origin(
            st.session_state["origin_input"] - timedelta(days=7)
        )
        st.rerun()
with col_next:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Week →", key="next_wk"):
        st.session_state["origin_input"] = _clamp_origin(
            st.session_state["origin_input"] + timedelta(days=7)
        )
        st.rerun()
with col_dnext:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Day →", key="next_day"):
        st.session_state["origin_input"] = _clamp_origin(
            st.session_state["origin_input"] + timedelta(days=1)
        )
        st.rerun()
with col_date:
    origin = st.date_input(
        "Day 0 (last known data)",
        min_value=date(2017, 2, 1),
        max_value=max_date,
        help="The model uses everything up to and including this date, then forecasts the next 7 days.",
        key="origin_input",
    )
with col_ctx:
    if origin in actuals:
        st.metric(
            label=f"Day 0: {origin.strftime('%A')} {origin.isoformat()}",
            value=f"{actuals[origin]:.0f} trips (known)",
        )
    st.caption("Try: **Oct 5**, **Jun 22** (good) · **Sep 7** (strike) · **Jan 12** (winter)")

df = run_forecast(origin, 7)

if not df.empty:
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

        # Forecast — label each point as Day 1..7
        day_labels = [f"Day {h}" for h in df["horizon"]]
        fig.add_trace(go.Scatter(
            x=df["target"], y=df["predicted"],
            mode="lines+markers+text", name="Forecast",
            line=dict(color="#1f77b4", width=3), marker=dict(size=10),
            text=day_labels,
            textposition="top center",
            textfont=dict(size=10, color="#1f77b4"),
            customdata=list(zip(
                df["lower"].round(0), df["upper"].round(0),
                df["horizon"],
                [f"Day {h}" for h in df["horizon"]],
            )),
            hovertemplate=(
                "<b>%{customdata[3]}</b> — %{x|%a %b %d}<br>"
                "Forecast: %{y:.0f} trips<br>"
                "80%% interval: [%{customdata[0]:.0f} .. %{customdata[1]:.0f}]<extra></extra>"
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
                customdata=[f"Day {h}" for h in kn["horizon"]],
                hovertemplate=(
                    "<b>%{customdata}</b> — %{x|%a %b %d}<br>"
                    "Actual: %{y:.0f} trips<extra></extra>"
                ),
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
        st.caption(
            f"**All 7 predictions are made at the same time from Day 0 "
            f"({origin.strftime('%a %b %d')}).** "
            f"Day 1 (tomorrow) is the most accurate because the model has the freshest data. "
            f"Day 7 is the least accurate. Use ← Day / Day → to shift by one day and see how "
            f"the forecast window moves."
        )

# Summary KPIs
if not df.empty:
    if df["actual"].notna().any():
        kn = df.dropna(subset=["actual"])
        in_iv = (kn["actual"] >= kn["lower"]) & (kn["actual"] <= kn["upper"])
        mae = (kn["predicted"] - kn["actual"]).abs().mean()
        wape = (kn["predicted"] - kn["actual"]).abs().sum() / max(kn["actual"].abs().sum(), 1e-8)
        c1, c2, c3 = st.columns(3)
        c1.metric("Days inside interval", f"{int(in_iv.sum())} of {len(kn)}")
        c2.metric("Avg error (MAE)", f"{mae:.0f} trips/day")
        c3.metric("This week's WAPE", f"{wape:.1%}")

    # ==================================================================
    # AI EXPLANATION — full-width, gradient-bordered, streaming
    # ==================================================================
    # CSS for the Apple-Intelligence-style panel
    st.markdown(
        """
        <style>
        .ai-panel {
            position: relative;
            padding: 28px 32px;
            border-radius: 18px;
            margin-top: 24px;
            background: linear-gradient(135deg, #ffffff 0%, #fafbff 100%);
            box-shadow:
                0 1px 3px rgba(102, 126, 234, 0.06),
                0 12px 40px -12px rgba(118, 75, 162, 0.12);
        }
        .ai-panel::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 18px;
            padding: 2px;
            background: linear-gradient(
                135deg,
                #667eea 0%,
                #9f7aea 25%,
                #ed64a6 50%,
                #f56565 75%,
                #ed8936 100%
            );
            -webkit-mask:
                linear-gradient(#fff 0 0) content-box,
                linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
            opacity: 0.85;
        }
        .ai-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 4px;
            font-size: 1.05rem;
            font-weight: 600;
            background: linear-gradient(90deg, #667eea, #9f7aea, #ed64a6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .ai-subtitle {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 18px;
        }
        .ai-content h3,
        .ai-content h4 {
            margin-top: 1.1em;
            margin-bottom: 0.4em;
            font-weight: 600;
            color: #1a1a2e;
        }
        .ai-content ul {
            margin-top: 0.3em;
            padding-left: 1.3em;
        }
        .ai-content li {
            margin-bottom: 0.4em;
            line-height: 1.55;
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        .ai-loading {
            background: linear-gradient(
                90deg,
                rgba(102, 126, 234, 0.08),
                rgba(237, 100, 166, 0.15),
                rgba(102, 126, 234, 0.08)
            );
            background-size: 200% 100%;
            animation: shimmer 2.2s infinite linear;
            height: 10px;
            border-radius: 5px;
            margin-top: 12px;
        }
        /* Target primary buttons with every reasonable selector so the style lands
           regardless of Streamlit's exact DOM version */
        button[kind="primary"],
        .stButton button[kind="primary"],
        div[data-testid="stButton"] button[kind="primary"],
        [data-testid="stBaseButton-primary"] {
            border-radius: 14px !important;
            border: none !important;
            background: linear-gradient(135deg, #667eea 0%, #9f7aea 40%, #ed64a6 80%, #f56565 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            padding: 14px 28px !important;
            min-height: 52px !important;
            box-shadow:
                0 4px 18px rgba(102, 126, 234, 0.32),
                0 1px 3px rgba(237, 100, 166, 0.18) !important;
            transition: all 0.22s ease !important;
            letter-spacing: 0.01em !important;
        }
        button[kind="primary"]:hover,
        .stButton button[kind="primary"]:hover,
        div[data-testid="stButton"] button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow:
                0 8px 24px rgba(102, 126, 234, 0.42),
                0 2px 6px rgba(237, 100, 166, 0.25) !important;
            filter: brightness(1.05) !important;
        }
        button[kind="primary"] p,
        [data-testid="stBaseButton-primary"] p {
            font-size: 1.05rem !important;
            font-weight: 700 !important;
        }
        .explain-hint {
            text-align: center;
            color: #6b7280;
            font-size: 0.88rem;
            margin-top: 8px;
            margin-bottom: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    # Button — with spacing and hint
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        explain_clicked = st.button(
            "✨  Explain this forecast with AI",
            type="primary",
            width="stretch",
            help="Streams a plain-English breakdown of what drove the model's predictions this week.",
        )
        st.markdown(
            "<p class='explain-hint'>Uses Claude to analyse the week's features and explain the forecast.</p>",
            unsafe_allow_html=True,
        )

    if explain_clicked:
        api_key = ""
        if hasattr(st, "secrets"):
            try:
                api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            except Exception:
                api_key = ""
        if not api_key:
            api_key = st.session_state.get("_anthropic_key", "")

        if not api_key:
            with st.expander("🔑 Enter Anthropic API key (not stored)"):
                k = st.text_input("API key", type="password", key="_key_input")
                if k:
                    st.session_state["_anthropic_key"] = k
                    api_key = k

        if api_key:
            import anthropic

            # Build rich context for the model
            week_data = []
            for _, r in df.iterrows():
                day = {
                    "date": r["target"].strftime("%A %Y-%m-%d"),
                    "horizon_days_ahead": int(r["horizon"]),
                    "predicted_trips": round(float(r["predicted"])),
                    "interval_lower_80pct": round(float(r["lower"])),
                    "interval_upper_80pct": round(float(r["upper"])),
                    "actual_trips": round(float(r["actual"])) if pd.notna(r["actual"]) else None,
                    "temperature_c": round(float(r["temp_mean_c"]), 1) if pd.notna(r["temp_mean_c"]) else None,
                    "rainfall_mm": round(float(r["precipitation_total_mm"]), 1) if pd.notna(r["precipitation_total_mm"]) else None,
                    "bank_holiday": bool(r["is_bank_holiday"]),
                    "tube_strike_flagged": bool(r["is_tube_strike"]),
                }
                if day["actual_trips"] is not None:
                    err = day["predicted_trips"] - day["actual_trips"]
                    day["error_trips"] = err
                    day["error_pct"] = round(err / max(day["actual_trips"], 1) * 100, 1)
                    day["inside_interval"] = (
                        day["actual_trips"] >= day["interval_lower_80pct"]
                        and day["actual_trips"] <= day["interval_upper_80pct"]
                    )
                week_data.append(day)

            origin_actual = actuals.get(origin, 0)

            prompt = f"""You are explaining a demand forecasting model's reasoning for one week of predictions.

## Model context

Target: daily Santander Cycles trips at Soho, London docking stations.
Training: 2017-2024 (excluding COVID 2020-2021) on 2,264 daily observations.
Architecture: three LightGBM quantile models (5th/50th/95th percentile) plus a per-horizon conformal buffer.

Top input features (by importance): temperature change day-over-day, wind speed, minimum temperature,
temperature anomaly vs 30-day average, 28-day rolling demand mean, holiday proximity, precipitation,
day-of-year seasonality, 1/7/14/28/365-day lags, daylight hours.

## What the model CANNOT see

- Live tube disruptions beyond the hand-curated strike list (many real-world disruptions aren't flagged)
- Real-time TfL bike availability / pricing / promotions
- Santander fleet expansion (new e-bikes, docking station additions)
- Hyperlocal events (individual shows, protests, filming closures)
- Weather extremes very poorly represented in training (single hottest day, deep freezes)
- Social effects (TikTok trends, viral events)
- Covid recovery effects (2020-2021 were excluded so model thinks 2022+ is "normal")

## Forecast origin

{origin.strftime('%A %Y-%m-%d')} — {origin_actual:.0f} actual trips.

## Week being explained

```json
{json.dumps(week_data, indent=2)}
```

## Your task

Produce a CONCISE, BULLET-POINT explanation using exactly these three markdown sections.

### 🎯 What drove the prediction this week

- 3-4 bullets naming the specific feature values in the data above that the model would have
  leaned on, and the direction they pushed the forecast.
- Reference EXACT numbers from the JSON: temperatures, rainfall, day-of-week, flag values.
- Do NOT say "the model was uncertain" or "the model might have considered X" — state which
  feature value caused which direction of adjustment.

### 📊 Per-day reasoning

- One bullet per day. For each day, name the 1-2 feature values from the data above that
  most likely set the prediction level relative to the week's average.
- Stick to fields actually provided: temperature_c, rainfall_mm, day-of-week,
  bank_holiday, tube_strike_flagged.

### ⚠️ Where the model missed

- ONE short bullet per day that was outside the 80% interval (skip days inside).
- Format: "[Day, date]: predicted X, actual Y. [reason]."
- For `[reason]`: if a feature value in the data plausibly explains the direction
  of the miss, state it in a clause (e.g., "model was high because temp was 18°C"). If no
  feature explains it, add a single-clause guess at the category only — "probably a local
  event", "probably an untracked transport issue", "unexplained by visible inputs".
- Do NOT invent specific events (no "possible concert", no "rumoured strike"). Category only.
- If every day was inside the interval, write just: "All days fell inside the 80% interval."

## Hard rules

- Bullet points only, no paragraphs.
- Each bullet: one sentence, two max.
- Quote EXACT numbers from the JSON. Don't round casually.
- Do NOT speculate about events, rumours, disruptions, or weather anomalies that aren't
  present in the provided data. If the JSON doesn't show it, don't claim it.
- Do NOT re-describe the chart or re-state the predictions/actuals as a summary.
- If you run out of reasons, stop. Do not pad.
"""

            panel_placeholder = st.empty()
            panel_placeholder.markdown(
                '<div class="ai-panel">'
                '<div class="ai-header">✨ Model reasoning</div>'
                '<div class="ai-subtitle">Generating explanation based on this week\'s inputs and outputs…</div>'
                '<div class="ai-loading"></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            try:
                import markdown as md_lib
                client = anthropic.Anthropic(api_key=api_key)
                week_label = df["target"].min().strftime("%a %b %d %Y")
                with client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1200,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    buf = []
                    for text in stream.text_stream:
                        buf.append(text)
                        # Convert streamed markdown -> HTML so headings/bullets render inside the panel
                        rendered = md_lib.markdown(
                            "".join(buf),
                            extensions=["fenced_code", "nl2br", "sane_lists"],
                        )
                        panel_placeholder.markdown(
                            f'<div class="ai-panel">'
                            f'<div class="ai-header">✨ Model reasoning</div>'
                            f'<div class="ai-subtitle">Forecast for the week starting {week_label}</div>'
                            f'<div class="ai-content">{rendered}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                # Persist so the panel survives reruns, and remember which week it was for
                st.session_state["_explanation_html"] = rendered
                st.session_state["_explanation_for"] = origin.isoformat()
                st.session_state["_explanation_week_label"] = week_label
            except Exception as e:
                panel_placeholder.markdown(
                    f'<div class="ai-panel">'
                    f'<div class="ai-header">✨ Model reasoning</div>'
                    f'<div class="ai-content">⚠️ API error: {e}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # If a previous explanation exists for this week and the user hasn't regenerated,
    # keep showing it with a hide button.
    elif (
        "_explanation_html" in st.session_state
        and st.session_state.get("_explanation_for") == origin.isoformat()
    ):
        persisted_label = st.session_state.get("_explanation_week_label", "")
        persisted_html = st.session_state["_explanation_html"]
        st.markdown(
            f'<div class="ai-panel">'
            f'<div class="ai-header">✨ Model reasoning</div>'
            f'<div class="ai-subtitle">Forecast for the week starting {persisted_label}</div>'
            f'<div class="ai-content">{persisted_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        hide_col1, hide_col2, hide_col3 = st.columns([1, 2, 1])
        with hide_col2:
            if st.button("✕ Hide explanation", key="hide_explanation", width="stretch"):
                for k in ("_explanation_html", "_explanation_for", "_explanation_week_label"):
                    st.session_state.pop(k, None)
                st.rerun()

st.divider()


# ==========================================================================
# ==========================================================================
# Tabs: Performance | Under the hood | What's missing
# ==========================================================================
tab_perf, tab_under, tab_missing = st.tabs([
    "📈 Performance",
    "🧠 Under the hood",
    "🔍 What's missing",
])

with tab_perf:
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
        st.caption("Actuals fall inside the interval ~80% of the time at every horizon — that's the target the interval is calibrated to hit.")

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
    **Why is winter less accurate?** Two things add up:

    1. **Low volume amplifies percentage error.** January averages ~680 trips/day vs ~1,800
       in June. Even the same absolute miss looks bigger as a percentage of a smaller number.

    2. **Absolute error is also higher in January.** Daily MAE is ~150 trips in January
       vs ~120 in June. The first weeks of 2025 had tube strikes and a cold/wet spell that
       caused bigger day-to-day swings. The model can see the weather but not everything
       else that drives winter demand — local events, untracked transport issues,
       post-holiday behaviour — so more of the variance is unexplained by its inputs.

    All WAPE numbers on this page use the same computation: `sum(|predicted - actual|) / sum(actual)`,
    on **1-day-ahead** predictions only (the most accurate horizon).
    """
    )

    st.divider()


    # ==========================================================================

with tab_under:
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
            "☀️ **Daylight** (3)": "Sunrise hour, sunset hour, daylight hours — calculated from "
                                    "the date and London's latitude. Always exact, never forecast.",
            "🔁 **Recent trips** (9)": "Trip count from yesterday, 1 week ago, 2 weeks ago, 4 weeks "
                                         "ago and 1 year ago, plus 7-day and 28-day rolling averages, "
                                         "plus a 7-day volatility measure.",
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

    st.header("How it works")
    st.markdown(
        """
    **What we predict.** The number of Santander Cycles trips per day at Soho-area docking
    stations (Moor Street, Wardour Street, Broadwick Street, Golden Square and nearby West
    End stations). 4.25 million trips since 2017, all from TfL's public data bucket.

    **The model.** Three LightGBM tree-boosting models working together — one predicting
    the low end (5th percentile), one the middle (50th), one the high end (95th). Each is
    an ensemble of ~500 decision trees.

    **The prediction interval.** The raw [low, high] gap from the models is often too
    narrow. We measure how often actuals fall outside it on held-out data and widen the
    interval by exactly the amount needed to cover 80% of actuals. That widening comes
    directly from observed errors — no hand-tuning.
    """
    )

    st.divider()


    # ==========================================================================

with tab_missing:
    st.header("What's missing from the model")
    st.markdown(
        "These are inputs that would improve accuracy but aren't included — "
        "either because no free historical data exists, or it would require "
        "ongoing manual curation."
    )
    st.markdown(
        """
    | Missing input | Why it matters | Why it's not included |
    |---|---|---|
    | **Live TfL disruption feed** | Tube disruptions push people to bikes (or deter travel entirely). The September 2025 week-long RMT strike caused a 2× surge we missed. | No free historical archive of unplanned disruptions. Our static strike list captures major actions only. |
    | **Hyperlocal events** | Film premieres, Chinese New Year, protest marches, fashion week pop-ups all affect Soho foot traffic. | No structured historical event data for this level of granularity. Would require ongoing manual curation or an events API. |
    | **Santander fleet size / pricing** | TfL's e-bike fleet grew significantly between 2024-2025, causing a structural demand uplift the model didn't anticipate. | TfL announces changes in press releases but doesn't publish a machine-readable time series of fleet size or pricing by date — so we can't use it as a training feature. |
    | **Real-time weather forecast** | The demo uses actual historical weather (perfect hindsight). In production, multi-day weather forecasts degrade past day 3-4 — the model would inherit that forecast error. | The API exists (Open-Meteo Forecast) and a client is built (`src/data/weather_forecast.py`), but the demo evaluates on 2025 using archive weather for a fair like-for-like comparison with training. |
    | **Social / viral events** | A TikTok trend or cycling campaign could temporarily spike demand. | No structured data source exists. |
    | **Competitor transport changes** | Lime / Uber bike availability, e-scooter regulation changes affect Santander usage. | No historical archive. |
    """
    )


st.caption(
    "Code: github.com/tmbellau/restaurant-test • "
    "Built with LightGBM, Plotly, Streamlit. "
    "All data sources are free and public."
)
