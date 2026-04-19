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


def run_convergence(target_date: date) -> pd.DataFrame:
    """Predict ONE target date from 7 different origins (h=7d back to h=1d).

    Shows how the prediction converges on the actual as fresh data arrives.
    """
    daily, idx, actuals = load_daily()
    models, features, buffers = load_models()
    if target_date not in idx:
        return pd.DataFrame()
    rows = []
    for h in range(7, 0, -1):
        origin = target_date - timedelta(days=h)
        if origin not in actuals:
            continue
        row = daily.iloc[idx[target_date]].copy()
        row = recompute_lags(row, target_date, origin, actuals)
        q = predict_one(models, features, row, horizon=h, buffers=buffers)
        rows.append({
            "origin": origin,
            "horizon": h,
            "origin_dow": origin.strftime("%a"),
            "lower": q["lower"],
            "predicted": q["predicted"],
            "upper": q["upper"],
            "interval_width": q["upper"] - q["lower"],
            "horizon_buffer": q["horizon_buffer"],
        })
    df = pd.DataFrame(rows)
    return df


# ----- Main -----
st.title("🎯 Demand forecast")

_, idx, actuals = load_daily()
min_date = min(actuals.keys())
max_date = max(actuals.keys())

tab_forecast, tab_convergence = st.tabs([
    "📅 7-day forecast from an origin",
    "🔍 Convergence: how a prediction improves",
])


# ===================================================================
# TAB 1: 7-day forecast from a single origin (existing functionality)
# ===================================================================
with tab_forecast:
    st.markdown(
        "Pick an **origin date** and the model forecasts the next 7 days. "
        "Intervals widen at longer horizons because the model is less certain."
    )

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        origin = st.date_input(
            "Origin date (the last known actual)",
            value=date(2025, 6, 15),
            min_value=date(2017, 2, 1),
            max_value=max_date,
            help="The model predicts the 7 days AFTER this date.",
            key="origin_tab1",
        )
    with col2:
        days = st.slider("Forecast horizon (days)", 1, 7, 7, key="days_tab1")
    with col3:
        st.markdown("**Origin context**")
        if origin in actuals:
            st.metric(
                label=f"{origin.strftime('%A')} {origin.isoformat()}",
                value=f"{actuals[origin]:.0f} trips",
            )
        else:
            st.info(f"{origin}: no actual available")

    df = run_forecast(origin, days)

    if df.empty:
        st.warning("No forecast could be built — check that the origin date is within the data range.")
    else:
        # Chart
        fig = go.Figure()
        hist_dates = [origin - timedelta(days=k) for k in range(14, -1, -1)]
        hist_actuals = [actuals.get(d, None) for d in hist_dates]
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_actuals,
            mode="lines+markers", name="Past actuals",
            line=dict(color="#888", width=2), marker=dict(size=6),
            hovertemplate="<b>%{x|%a %b %d}</b><br>Actual: %{y:.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=list(df["target"]) + list(df["target"])[::-1],
            y=list(df["upper"]) + list(df["lower"])[::-1],
            fill="toself", fillcolor="rgba(31,119,180,0.15)",
            line=dict(color="rgba(31,119,180,0)"),
            name="80% interval", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df["target"], y=df["predicted"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#1f77b4", width=3), marker=dict(size=10),
            customdata=list(zip(df["lower"].round(0), df["upper"].round(0), df["horizon"])),
            hovertemplate=(
                "<b>%{x|%a %b %d}</b> (h=%{customdata[2]}d)<br>"
                "Forecast: %{y:.0f}<br>"
                "Interval: [%{customdata[0]:.0f} .. %{customdata[1]:.0f}]<extra></extra>"
            ),
        ))
        if df["actual"].notna().any():
            known = df[df["actual"].notna()]
            in_iv = (known["actual"] >= known["lower"]) & (known["actual"] <= known["upper"])
            colors = ["#2ca02c" if b else "#d62728" for b in in_iv]
            fig.add_trace(go.Scatter(
                x=known["target"], y=known["actual"],
                mode="markers", name="Actual",
                marker=dict(size=14, color=colors, symbol="diamond",
                            line=dict(color="white", width=1.5)),
                hovertemplate="<b>%{x|%a %b %d}</b><br>Actual: %{y:.0f}<extra></extra>",
            ))
        fig.add_vline(x=origin, line_width=1.5, line_dash="dot", line_color="#555")
        fig.add_annotation(x=origin, y=1.02, yref="paper", text="origin",
                           showarrow=False, font=dict(size=11, color="#555"))
        fig.update_layout(
            height=460, hovermode="x unified",
            xaxis_title=None, yaxis_title="Daily trips at Soho stations",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        st.plotly_chart(fig, width="stretch")

        # Table
        def _fmt(x): return f"{x:.0f}" if pd.notna(x) else "—"
        def _flags(r):
            fl = []
            if r["is_bank_holiday"]: fl.append("🏖 hol")
            if r["is_tube_strike"]: fl.append("🚇 strike")
            if r["is_cultural_period"]: fl.append("🎉 event")
            return ", ".join(fl)
        display = pd.DataFrame({
            "Date": df["target"].dt.strftime("%a %Y-%m-%d"),
            "h": df["horizon"].apply(lambda h: f"{h}d"),
            "Lower": df["lower"].apply(_fmt),
            "Forecast": df["predicted"].apply(_fmt),
            "Upper": df["upper"].apply(_fmt),
            "Width": df["interval_width"].apply(_fmt),
            "Actual": df["actual"].apply(_fmt),
            "Flags": df.apply(_flags, axis=1),
        })
        st.dataframe(display, width="stretch", hide_index=True)

        # KPIs
        if df["actual"].notna().any():
            kn = df.dropna(subset=["actual"])
            in_flags = (kn["actual"] >= kn["lower"]) & (kn["actual"] <= kn["upper"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Coverage", f"{int(in_flags.sum())}/{len(kn)} days in interval")
            c2.metric("MAE", f"{(kn['predicted'] - kn['actual']).abs().mean():.0f} trips")
            c3.metric("WAPE", f"{(kn['predicted'] - kn['actual']).abs().sum() / max(kn['actual'].abs().sum(), 1e-8):.1%}")


# ===================================================================
# TAB 2: Convergence — pick a target, see prediction improving
# ===================================================================
with tab_convergence:
    st.markdown(
        "Pick a **target date** and see how the model's prediction for that "
        "day **improves as the forecast horizon shrinks** — from 7 days out "
        "(earliest, least accurate) to 1 day out (latest, most accurate). "
        "Each row is an independent forecast made from a different origin date."
    )

    col1, col2 = st.columns([2, 3])
    with col1:
        target_date = st.date_input(
            "Target date to predict",
            value=date(2025, 6, 20),
            min_value=date(2017, 2, 8),
            max_value=max_date,
            help="The model predicts THIS date from 7 different starting points.",
            key="target_tab2",
        )
    with col2:
        actual_val = actuals.get(target_date, None)
        if actual_val is not None:
            st.metric(
                label=f"Actual for {target_date.strftime('%A')} {target_date.isoformat()}",
                value=f"{actual_val:.0f} trips",
            )
        else:
            st.info(f"No actual available for {target_date}")

    conv_df = run_convergence(target_date)
    if conv_df.empty:
        st.warning("No convergence data — target date may be outside the data range.")
    else:
        # Convergence chart
        fig = go.Figure()

        # 80% interval band (per-horizon — should visibly narrow)
        fig.add_trace(go.Scatter(
            x=list(conv_df["horizon"]) + list(conv_df["horizon"])[::-1],
            y=list(conv_df["upper"]) + list(conv_df["lower"])[::-1],
            fill="toself", fillcolor="rgba(31,119,180,0.15)",
            line=dict(color="rgba(31,119,180,0)"),
            name="80% interval", hoverinfo="skip",
        ))

        # Predictions line
        fig.add_trace(go.Scatter(
            x=conv_df["horizon"], y=conv_df["predicted"],
            mode="lines+markers", name="Prediction",
            line=dict(color="#1f77b4", width=3), marker=dict(size=12),
            customdata=list(zip(
                conv_df["lower"].round(0), conv_df["upper"].round(0),
                conv_df["origin"].apply(lambda d: d.strftime("%a %b %d")),
                conv_df["interval_width"].round(0),
            )),
            hovertemplate=(
                "<b>%{customdata[2]}</b> (h=%{x}d)<br>"
                "Prediction: %{y:.0f}<br>"
                "Interval: [%{customdata[0]:.0f} .. %{customdata[1]:.0f}] "
                "(width %{customdata[3]:.0f})<extra></extra>"
            ),
        ))

        # Actual horizontal line
        if actual_val is not None:
            fig.add_hline(
                y=actual_val, line_dash="dash", line_color="#2ca02c", line_width=2,
                annotation_text=f"Actual: {actual_val:.0f}",
                annotation_position="top right",
                annotation_font=dict(color="#2ca02c", size=13),
            )

        fig.update_layout(
            height=460,
            xaxis=dict(
                title="Days before target (7 = earliest forecast, 1 = most recent)",
                tickvals=list(range(1, 8)),
                ticktext=[f"{h}d out" for h in range(1, 8)],
                autorange="reversed",
            ),
            yaxis_title="Predicted daily trips",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        st.plotly_chart(fig, width="stretch")

        # Detail table
        ct = pd.DataFrame({
            "Origin": conv_df["origin"].apply(lambda d: d.strftime("%a %Y-%m-%d")),
            "Horizon": conv_df["horizon"].apply(lambda h: f"{h} day{'s' if h > 1 else ''} out"),
            "Lower": conv_df["lower"].apply(lambda x: f"{x:.0f}"),
            "Prediction": conv_df["predicted"].apply(lambda x: f"{x:.0f}"),
            "Upper": conv_df["upper"].apply(lambda x: f"{x:.0f}"),
            "Width": conv_df["interval_width"].apply(lambda x: f"{x:.0f}"),
        })
        if actual_val is not None:
            ct["Actual"] = f"{actual_val:.0f}"
            ct["Error"] = conv_df["predicted"].apply(lambda p: f"{p - actual_val:+.0f}")
        st.dataframe(ct, width="stretch", hide_index=True)

        if actual_val is not None:
            errors = (conv_df["predicted"] - actual_val).abs()
            st.caption(
                f"Prediction error shrinks from **{errors.iloc[0]:.0f}** trips "
                f"(7d out) to **{errors.iloc[-1]:.0f}** trips (1d out) — "
                f"a {(1 - errors.iloc[-1]/max(errors.iloc[0], 1)):.0%} improvement."
            )


st.markdown("---")
with st.expander("💡 How to read this"):
    st.markdown("""
**7-day forecast tab:**
- **Grey line** = 2 weeks of past actuals — the context the model sees.
- **Blue line** = the point forecast (50th percentile).
- **Blue band** = the 80% prediction interval, **wider at longer horizons** because the model is less confident about further-out days.
- **Green / red diamonds** = actual values inside / outside the interval.

**Convergence tab:**
- Shows a **single target date** predicted 7 times — from 7 days out (most uncertain) down to 1 day out (most certain).
- **Green dashed line** = the actual value.
- Watch the blue prediction line **converge** toward the actual as the horizon shrinks.
- The **interval narrows** as the forecast gets closer — less uncertainty, more data.
""")
