"""Shared feature engineering for the freight-rate model."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_LEVELS = ["Dry Van", "Reefer", "Flatbed"]


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues without dropping rows."""
    out = df.copy()
    # Sign-entry error: a small number of weights are recorded negative.
    # Magnitudes are in-range, so we take the absolute value instead of dropping.
    out["weight"] = out["weight"].abs()
    return out


def build_features(df: pd.DataFrame, weight_median: float, market_median: float) -> pd.DataFrame:
    out = clean_raw(df)
    out["date"] = pd.to_datetime(out["date"])

    # Missing-value handling: weight/market_index are missing at low rates (<1%)
    # and show near-zero correlation with rate, so median imputation (fit on
    # training data only) is used, with indicator flags kept for transparency.
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["weight"] = out["weight"].fillna(weight_median)
    out["market_index"] = out["market_index"].fillna(market_median)

    out["log_distance"] = np.log(out["distance"])
    out["calc_distance"] = haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["route_circuity"] = out["distance"] / out["calc_distance"].clip(lower=1)

    out["day_of_year"] = out["date"].dt.dayofyear
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek

    for lvl in EQUIPMENT_LEVELS:
        out[f"equip_{lvl.replace(' ', '_')}"] = (out["equipment"] == lvl).astype(int)

    feature_cols = [
        "log_distance",
        "distance",
        "calc_distance",
        "route_circuity",
        "weight",
        "weight_missing",
        "market_index",
        "market_index_missing",
        "quote_signal",
        "day_of_year",
        "month",
        "day_of_week",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ] + [f"equip_{lvl.replace(' ', '_')}" for lvl in EQUIPMENT_LEVELS]

    return out, feature_cols
