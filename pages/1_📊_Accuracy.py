"""Accuracy page — full 2025 holdout evaluation.

Every day in 2025 was predicted 7 times (at horizons 1-7 days out). This
page visualizes how the model performed across the full year.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Accuracy — Soho Cycles Forecast", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def load_eval() -> pd.DataFrame:
    df = pd.read_parquet("data/training/eval_2025.parquet")
    df["target"] = pd.to_datetime(df["target"])
    return df


with st.sidebar:
    st.title("🚲 Soho Cycles")
    st.markdown("- 🎯 Predict")
    st.markdown("- 📊 **Accuracy** (this page)")
    st.markdown("- ⚙️ How it works")
    st.markdown("---")
    st.caption("All metrics below come from the 2025 holdout — "
               "the model was trained on 2017-2024 data only.")


st.title("📊 Accuracy on unseen data")
st.markdown(
    "The model was trained only on 2017-2024 data (ex-COVID). Every day in 2025 "
    "was held out and predicted at every horizon from 1 to 7 days ahead. "
    "Here's how it did."
)

df = load_eval()
df_ok = df[~df["is_outage"]].copy()  # exclude ~2 data outage days

# ============ Top KPI row ============
col1, col2, col3, col4 = st.columns(4)

overall_cov = df_ok["in_interval"].mean()
overall_wape = df_ok["abs_error"].sum() / df_ok["actual"].abs().sum()
avg_actual = df_ok["actual"].mean()
median_pct = df_ok["pct_error"].abs().median()

col1.metric("80% interval coverage", f"{overall_cov:.1%}", delta=f"{(overall_cov-0.80)*100:+.1f}pp vs target")
col2.metric("WAPE (all horizons)", f"{overall_wape:.1%}")
col3.metric("Median % error", f"{median_pct:.1f}%")
col4.metric("Avg daily trips", f"{avg_actual:.0f}")

st.markdown("---")


# ============ Accuracy by horizon ============
st.subheader("Accuracy by forecast horizon")
st.caption("As the forecast horizon shrinks, the most-recent actual becomes "
           "the `covers_1d_lag` feature — fresher and more informative. "
           "Longer-horizon forecasts lean more on weather and calendar.")

horizon_stats = df_ok.groupby("horizon").agg(
    coverage=("in_interval", "mean"),
    wape=("abs_error", lambda x: x.sum() / df_ok.loc[x.index, "actual"].abs().sum()),
    mae=("abs_error", "mean"),
    interval_width=("upper", lambda u: (u - df_ok.loc[u.index, "lower"]).mean()),
).reset_index()

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=horizon_stats["horizon"].astype(str) + "d",
        y=(horizon_stats["wape"] * 100).round(2),
        marker_color=["#1f77b4"] * len(horizon_stats),
        text=(horizon_stats["wape"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        name="WAPE",
    ))
    fig.update_layout(
        title="WAPE by horizon",
        yaxis_title="WAPE (%)",
        xaxis_title="Forecast horizon",
        height=360,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, max(horizon_stats["wape"] * 100) * 1.2])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=horizon_stats["horizon"].astype(str) + "d",
        y=(horizon_stats["coverage"] * 100).round(1),
        marker_color=["#2ca02c" if c >= 0.77 else "#ff7f0e"
                      for c in horizon_stats["coverage"]],
        text=(horizon_stats["coverage"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        name="Coverage",
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="#888",
                  annotation_text="80% target", annotation_position="top right")
    fig.update_layout(
        title="Prediction interval coverage by horizon",
        yaxis_title="Coverage (%)",
        xaxis_title="Forecast horizon",
        height=360,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


# ============ Full year line chart ============
st.subheader("Full year 2025: forecast vs actual (1-day-ahead)")
st.caption("Every day predicted 1 day ahead. Green diamonds = actual fell inside "
           "the 80% interval. Red diamonds = actual fell outside.")

h1 = df_ok[df_ok["horizon"] == 1].sort_values("target").copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(h1["target"]) + list(h1["target"])[::-1],
    y=list(h1["upper"]) + list(h1["lower"])[::-1],
    fill="toself",
    fillcolor="rgba(31,119,180,0.10)",
    line=dict(color="rgba(31,119,180,0)"),
    name="80% interval",
    hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=h1["target"], y=h1["predicted"],
    mode="lines",
    line=dict(color="#1f77b4", width=1.4),
    name="Forecast",
    hovertemplate="<b>%{x|%a %b %d}</b><br>Predicted: %{y:.0f}<extra></extra>",
))
colors = ["#2ca02c" if b else "#d62728" for b in h1["in_interval"]]
fig.add_trace(go.Scatter(
    x=h1["target"], y=h1["actual"],
    mode="markers",
    marker=dict(size=6, color=colors, line=dict(width=0)),
    name="Actual",
    hovertemplate="<b>%{x|%a %b %d}</b><br>Actual: %{y:.0f}<extra></extra>",
))
fig.update_layout(
    height=420,
    hovermode="x unified",
    xaxis_title=None,
    yaxis_title="Daily trips",
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)


# ============ Error scatter + distribution ============
st.subheader("Error analysis")
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h1["actual"], y=h1["predicted"],
        mode="markers",
        marker=dict(
            size=6, color=h1["pct_error"].abs(),
            colorscale="Viridis_r", showscale=True,
            colorbar=dict(title="|% error|"),
            opacity=0.7,
        ),
        hovertemplate=(
            "<b>%{customdata[0]|%a %b %d}</b><br>"
            "Actual: %{x:.0f}<br>Predicted: %{y:.0f}<br>"
            "Error: %{customdata[1]:+.0f}<extra></extra>"
        ),
        customdata=np.stack([h1["target"], h1["error"]], axis=-1),
    ))
    lo = min(h1["actual"].min(), h1["predicted"].min()) * 0.9
    hi = max(h1["actual"].max(), h1["predicted"].max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", line=dict(dash="dash", color="#888", width=1.5),
        name="Perfect",
        showlegend=False,
    ))
    fig.update_layout(
        title=f"Predicted vs actual (1-day-ahead, r = {np.corrcoef(h1['actual'], h1['predicted'])[0,1]:.3f})",
        xaxis_title="Actual daily trips",
        yaxis_title="Predicted daily trips",
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    pct = h1["pct_error"].replace([np.inf, -np.inf], np.nan).dropna()
    pct = pct[pct.abs() <= 80]  # trim a couple of extreme outliers for histogram visibility
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pct, nbinsx=40,
        marker_color="#1f77b4", marker_line_color="white", marker_line_width=0.5,
    ))
    fig.add_vline(x=0, line_width=1.2, line_color="black")
    fig.add_vline(x=float(pct.median()), line_dash="dash", line_color="#2ca02c",
                  annotation_text=f"median {pct.median():.1f}%",
                  annotation_position="top")
    within10 = (pct.abs() <= 10).mean() * 100
    within20 = (pct.abs() <= 20).mean() * 100
    fig.update_layout(
        title=f"% error distribution — {within10:.0f}% within ±10%, {within20:.0f}% within ±20%",
        xaxis_title="% error = (predicted − actual) / actual × 100",
        yaxis_title="Days",
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============ Monthly breakdown ============
st.subheader("Accuracy by month")

monthly = h1.copy()
monthly["month"] = monthly["target"].dt.month
monthly["month_name"] = monthly["target"].dt.month_name().str[:3]
month_stats = monthly.groupby(["month", "month_name"]).agg(
    wape=("abs_error", lambda x: x.sum() / monthly.loc[x.index, "actual"].abs().sum()),
    coverage=("in_interval", "mean"),
    n=("actual", "count"),
).reset_index().sort_values("month")

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(
        month_stats, x="month_name", y=(month_stats["wape"] * 100).round(2),
        title="WAPE by month (1-day-ahead)",
        color=(month_stats["wape"] * 100).round(2),
        color_continuous_scale="RdYlGn_r",
        labels={"y": "WAPE (%)", "month_name": "Month"},
    )
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0),
                      showlegend=False, coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        month_stats, x="month_name", y=(month_stats["coverage"] * 100).round(1),
        title="Coverage by month (1-day-ahead)",
        color=(month_stats["coverage"] * 100),
        color_continuous_scale="RdYlGn",
        labels={"y": "Coverage (%)", "month_name": "Month"},
    )
    fig.add_hline(y=80, line_dash="dash", line_color="#555")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0),
                      showlegend=False, coloraxis_showscale=False)
    fig.update_traces(texttemplate="%{y:.0f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.subheader("Browse individual predictions")
st.caption("Filter the full 2025 prediction record.")

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    selected_month = st.selectbox("Month", ["All"] + list(range(1, 13)))
with c2:
    selected_horizon = st.selectbox("Horizon", ["All"] + [f"{h}d" for h in range(1, 8)])
with c3:
    show_bad_only = st.checkbox("Only show misses (outside interval)", False)

filtered = df_ok.copy()
if selected_month != "All":
    filtered = filtered[filtered["target"].dt.month == int(selected_month)]
if selected_horizon != "All":
    filtered = filtered[filtered["horizon"] == int(selected_horizon.replace("d", ""))]
if show_bad_only:
    filtered = filtered[~filtered["in_interval"]]

display = filtered[["target", "dow", "horizon", "lower", "predicted", "upper",
                    "actual", "error", "pct_error", "in_interval"]].copy()
display["target"] = display["target"].dt.strftime("%Y-%m-%d")
display["lower"] = display["lower"].round(0).astype(int)
display["predicted"] = display["predicted"].round(0).astype(int)
display["upper"] = display["upper"].round(0).astype(int)
display["actual"] = display["actual"].round(0).astype(int)
display["error"] = display["error"].round(0).astype(int)
display["pct_error"] = display["pct_error"].round(1)
display["in_interval"] = display["in_interval"].map({True: "✓", False: "✗"})
display.columns = ["Date", "DoW", "h", "Lower", "Pred", "Upper", "Actual", "Err", "% Err", "In"]

st.dataframe(display, use_container_width=True, hide_index=True, height=400)
st.caption(f"{len(filtered)} rows shown")
