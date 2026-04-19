# Soho Cycles Demand Forecast

A demand-forecasting proof of concept: predict daily Santander Cycles trips
at Soho-area docking stations using only freely-available signals
(weather, UK calendar, daylight, tube strikes, lag history).

**80% interval coverage, 11% WAPE at 1-day-ahead**, validated on unseen 2025 data.

## Try the demo

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501 to explore:

- 🎯 **Predict** — pick any date, get a 7-day forecast with prediction
  intervals, compared to actuals where available.
- 📊 **Accuracy** — full 2025 holdout evaluation: WAPE by horizon,
  year-long line chart, predicted-vs-actual scatter, error distribution,
  monthly breakdown, browseable prediction records.
- ⚙️ **How it works** — features, training pipeline, data sources,
  feature importances, model architecture.

## Deploy to the web

The simplest path is **[Streamlit Community Cloud](https://share.streamlit.io/)** (free):

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select the repo, branch, and `streamlit_app.py` as
   the entry point.
4. Click **Deploy**. You get a public `*.streamlit.app` URL in ~2 minutes.

Every subsequent `git push` auto-redeploys.

Alternatives:
- **Hugging Face Spaces** (free, select "Streamlit" SDK)
- **Render.com** (free tier, add `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0` as the start command)
- **Fly.io** (Dockerfile deployment)

## Rebuilding the model from scratch

```bash
# Pull raw Santander data + weather + build features
python -m src.data.build_training_set --start 2017-01-01 --end 2025-12-31

# Aggregate to daily, add derived features, exclude COVID
python -m src.data.build_daily_training_set --exclude-covid

# Train the three quantile models + compute conformal buffer
python -m src.models.train_quantile_ensemble --train-end 2024-12-31

# Full 2025 holdout evaluation
python scripts/evaluate_2024.py
```

## Live forward prediction (when new actuals arrive)

```bash
# One shot: pull new data, rebuild, forecast next 7 days, score historical predictions
bash scripts/refresh_and_predict.sh
```

Predictions are stored in `data/predictions/` and compared against actuals
the next time the script runs.

## Key results (2025 holdout, 365 days)

| Horizon | WAPE | Coverage (80% target) |
|---|---|---|
| 1 day ahead | 10.7% | 87.1% |
| 3 days ahead | 12.8% | 78.0% |
| 7 days ahead | 12.8% | 79.1% |
| **Overall** | **12.6%** | **80.0%** |

- **58% of days within ±10% error, 83% within ±20%.**
- Correlation with actuals: **0.85.**
- Beats a naive "predict yesterday" baseline by 6-7 percentage points at
  every horizon.

## Honest limitations

- **Structural shifts** (e.g. a 30% YoY increase in Soho cycling in 2025,
  likely driven by TfL e-bike fleet expansion) can't be predicted from
  historical patterns and require periodic retraining.
- **Rare events** (tube strikes with full-network impact) are
  under-represented in training — ~25 strike days out of 2,200 is not
  enough for the model to learn a stable strike coefficient.
- **Data outages** (occasional days where TfL's station data returns
  near-zero counts) are handled with an origin-selection filter but still
  show up as "false negatives" in held-out evaluation.
- This is **not** a restaurant-demand forecasting model. It's a cycling
  demand model. The architecture transfers to any target with similar
  driver signals, but the proxy doesn't.
