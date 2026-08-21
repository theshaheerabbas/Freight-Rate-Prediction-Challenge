"""
Freight rate model: train, validate, and generate submission files.

Usage:
    python src/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA / "train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def time_split(df: pd.DataFrame, cutoff: str = "2025-09-01"):
    """Out-of-time split: train on earlier months, test on the most recent
    months actually present in train_test.csv. This mirrors the real task,
    where the model must extrapolate from Jan-Oct data to Nov-Dec predictions,
    rather than a random split that would understate error on unseen dates."""
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def evaluate(y_true_log, pred_log, label):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{label:14s}] MAE=${mae:8.2f}  RMSE=${rmse:8.2f}  MAPE={mape*100:5.2f}%  R2={r2:.4f}")
    return {"model": label, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def main():
    df = load_train()
    train_raw, test_raw = time_split(df)
    print(f"Train rows: {len(train_raw):,} (through {train_raw['date'].max().date()})")
    print(f"Held-out test rows: {len(test_raw):,} ({test_raw['date'].min().date()} to {test_raw['date'].max().date()})")

    # Imputation stats fit on the TRAIN split only (avoid leakage into test)
    weight_median = train_raw["weight"].abs().median()
    market_median = train_raw["market_index"].median()

    train_feat, feature_cols = build_features(train_raw, weight_median, market_median)
    test_feat, _ = build_features(test_raw, weight_median, market_median)

    X_train = train_feat[feature_cols].values
    X_test = test_feat[feature_cols].values
    y_train = np.log(train_raw["posted_rate"].values)
    y_test = np.log(test_raw["posted_rate"].values)

    results = []

    # Candidate 1: Ridge on standardized engineered features. The earlier EDA
    # (log-log OLS) showed log(rate) is ~94% explained by log(distance) plus
    # equipment/market/weight terms -- i.e. close to log-linear -- so a
    # regularized linear model is a natural, low-variance candidate that also
    # extrapolates smoothly into Nov/Dec, which lie outside the training window.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 3, 10, 30, 100])
    pred = fit_predict(ridge, X_train_s, y_train, X_test_s)
    results.append(evaluate(y_test, pred, "Ridge"))
    print(f"  -> RidgeCV selected alpha={ridge.alpha_}")

    # Candidate 2: Random Forest (captures non-linear interactions / thresholds)
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    )
    pred = fit_predict(rf, X_train, y_train, X_test)
    results.append(evaluate(y_test, pred, "RandomForest"))

    # Candidate 3: Gradient Boosting (also captures non-linear structure, usually
    # the strongest tabular learner)
    hgb = HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=6,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    pred = fit_predict(hgb, X_train, y_train, X_test)
    results.append(evaluate(y_test, pred, "HistGB"))

    pd.DataFrame(results).to_csv(OUT / "model_comparison.csv", index=False)

    # On the out-of-time held-out split (Sep-Oct, predicted from Jan-Aug), Ridge
    # matched or beat both tree ensembles on MAE/R2 -- confirming the relationship
    # is close to log-linear, and that the extra tree-model complexity does not
    # buy generalization here. Ridge is also safer for the Nov/Dec extrapolation
    # required by this task, since tree models cannot extrapolate a time trend
    # past the max date they were trained on. Final choice: Ridge (log-target).
    weight_median_full = df["weight"].abs().median()
    market_median_full = df["market_index"].median()
    full_feat, feature_cols = build_features(df, weight_median_full, market_median_full)
    X_full = full_feat[feature_cols].values
    y_full = np.log(df["posted_rate"].values)

    final_model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 1, 3, 10, 30, 100]))
    final_model.fit(X_full, y_full)
    print(f"Final RidgeCV alpha on full data: {final_model.named_steps['ridgecv'].alpha_}")

    # In-sample sanity check
    in_sample_pred = final_model.predict(X_full)
    evaluate(y_full, in_sample_pred, "Final(train)")

    # ---- Predict on validation.csv ----
    val = pd.read_csv(DATA / "validation.csv")
    val["date"] = pd.to_datetime(val["date"])
    val_feat, _ = build_features(val, weight_median_full, market_median_full)
    val_pred_log = final_model.predict(val_feat[feature_cols].values)
    val_pred = np.exp(val_pred_log)

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    assert list(template["load_id"]) == list(val["load_id"]), "load_id order mismatch"
    template["predicted_rate"] = np.round(val_pred, 2)
    template.to_csv(OUT / "validation_predictions.csv", index=False)
    print(f"Wrote {OUT / 'validation_predictions.csv'} ({len(template)} rows)")

    # ---- Predict on the fixed December lane ----
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    dec["date"] = pd.to_datetime(dec["date"])
    # December inputs don't include lat/lon -- attach them from known city coordinates
    # in train_test.csv (Lexington / Fort Wayne appear many times there).
    coord_lookup = pd.concat(
        [
            df[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
            ),
            df[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
            ),
        ]
    ).drop_duplicates(subset="city")
    coord_map = coord_lookup.set_index("city")[["lat", "lon"]]
    dec["pickup_lat"] = coord_map.loc[dec["pickup"], "lat"].values
    dec["pickup_lon"] = coord_map.loc[dec["pickup"], "lon"].values
    dec["delivery_lat"] = coord_map.loc[dec["delivery"], "lat"].values
    dec["delivery_lon"] = coord_map.loc[dec["delivery"], "lon"].values
    # December inputs also don't include market_index / quote_signal -- use the
    # training-set averages for the most recent month (Oct) as the best available
    # proxy for current market conditions on this lane.
    recent = df[df["date"] >= "2025-10-01"]
    dec["market_index"] = recent["market_index"].mean()
    dec["quote_signal"] = recent["quote_signal"].mean()

    dec_feat, _ = build_features(dec, weight_median_full, market_median_full)
    dec_pred_log = final_model.predict(dec_feat[feature_cols].values)
    dec["predicted_rate"] = np.round(np.exp(dec_pred_log), 2)
    dec_out = dec[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]].copy()
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")
    dec_out.to_csv(OUT / "december_chart_inputs_completed.csv", index=False)
    print(f"Wrote {OUT / 'december_chart_inputs_completed.csv'} ({len(dec_out)} rows)")

    # Feature importances for the report (permutation importance works for any
    # estimator, including the Ridge pipeline)
    try:
        from sklearn.inspection import permutation_importance

        perm = permutation_importance(
            final_model, X_full[:5000], y_full[:5000], n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
        )
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": perm.importances_mean}).sort_values(
            "importance", ascending=False
        )
        imp_df.to_csv(OUT / "feature_importance.csv", index=False)
        print(imp_df.head(10).to_string(index=False))
    except Exception as exc:  # pragma: no cover
        print("permutation importance skipped:", exc)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
