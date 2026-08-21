# Freight Rate Prediction — ML Engineer Assessment

Predicts `posted_rate` (USD) for truckload freight quotes from lane, equipment,
weight, date, and two market-signal features.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/train.py
```

This reads `data/train_test.csv`, `data/validation.csv`,
`data/validation_predictions_template.csv`, and `data/december_chart_inputs.csv`,
then writes:

- `outputs/validation_predictions.csv` — final predictions for all 12,000 validation loads
- `outputs/december_chart_inputs_completed.csv` — the December lane forecast
- `outputs/model_comparison.csv`, `outputs/metrics.json`, `outputs/feature_importance.csv`

## Score / generate the required chart

```bash
python score.py --predictions outputs/validation_predictions.csv \
                --december-predictions outputs/december_chart_inputs_completed.csv \
                --output-dir scorer_results
```

## Approach summary

See `report.docx` for the full write-up. In short:

- **Split**: out-of-time — train on Jan–Aug 2025, held out Sep–Oct 2025 — because
  the task itself is out-of-time (train_test.csv is Jan–Oct, validation.csv is
  Nov–Dec), so a random split would overstate accuracy.
- **Data quality**: ~0.6% of `weight` values were negative sign-entry errors
  (magnitudes in-range) → took `abs()`. ~0.6–0.8% of `weight`/`market_index`
  were missing → median imputation (fit on train only) with a missingness flag.
- **Features**: log(distance), haversine distance (cross-check + route
  circuity), equipment one-hot, weight, market_index, quote_signal, day-of-year/
  month/day-of-week. Pickup/delivery city identity was **not** used directly —
  8 cities in `validation.csv` never appear in `train_test.csv`, and city-level
  effects were small (~0.02 log-rate std) relative to the overall residual
  (~0.16), so lat/lon-based features generalize better.
- **Model**: Ridge regression on log(posted_rate), chosen over Random Forest
  and HistGradientBoosting after all three were compared on the Sep–Oct
  held-out split (Ridge had the lowest MAE and highest R²) — see
  `outputs/model_comparison.csv`.
