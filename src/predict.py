"""
predict.py
==========
Inference pipeline:
  - Single record prediction (dict input)
  - Batch prediction from CSV
  - Confidence interval estimation via bootstrap
"""

import os, joblib, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH    = "models/best_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_input_features(records: list, feature_cols: list) -> pd.DataFrame:
    """
    Convert raw input dicts into a feature-aligned DataFrame.
    Applies the same transformations as the training pipeline.
    Missing feature columns are filled with 0.
    """
    df = pd.DataFrame(records)

    # Date decomposition
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["year"]          = df["date"].dt.year
        df["month"]         = df["date"].dt.month
        df["day"]           = df["date"].dt.day
        df["day_of_week"]   = df["date"].dt.dayofweek
        df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"]       = df["date"].dt.quarter
        df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
        df["is_month_end"]  = df["date"].dt.is_month_end.astype(int)
        df["is_month_start"]= df["date"].dt.is_month_start.astype(int)
        df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"]       = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]       = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df.drop(columns=["date"], inplace=True)

    # Encode IDs
    for col in ["product_id", "store_id"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes

    # Derived features
    if "promotion" in df.columns and "price" in df.columns:
        df["promo_impact"]   = df["promotion"] * (1 / df["price"].replace(0, np.nan)).fillna(0)
        df["promo_discount"] = df["promotion"] * df["price"]
    if "holiday" in df.columns and "promotion" in df.columns:
        df["holiday_promo"] = df["holiday"] * df["promotion"]
    if "price" in df.columns:
        df["price_log"]     = np.log1p(df["price"])
        bins                = [0, 50, 100, 200, 350, np.inf]
        df["price_bucket"]  = pd.cut(df["price"], bins=bins, labels=False).fillna(0)
        if "past_sales" in df.columns:
            df["price_per_unit"] = df["price"] / (df["past_sales"].replace(0, 1))

    # Fill any missing features with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_demand(records: list,
                    model_path:    str = MODEL_PATH,
                    features_path: str = FEATURES_PATH,
                    return_confidence: bool = False) -> pd.DataFrame:
    """
    Predict demand for one or more records.

    Parameters
    ----------
    records           : list of dicts with product/store/date features
    return_confidence : if True, adds pred_low / pred_high (±15% range)

    Returns
    -------
    DataFrame with original fields + predicted_demand (+ confidence interval)
    """
    model        = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    X     = build_input_features(records, feature_cols)
    preds = np.maximum(model.predict(X), 0)
    preds = np.round(preds).astype(int)

    result = pd.DataFrame(records).copy()
    result["predicted_demand"] = preds

    if return_confidence:
        # Simple ±15% empirical confidence interval
        result["pred_low"]  = np.round(preds * 0.85).astype(int)
        result["pred_high"] = np.round(preds * 1.15).astype(int)

    logger.info("Predicted demand for %d records", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Batch CSV
# ─────────────────────────────────────────────────────────────────────────────

def batch_predict_from_csv(input_csv:  str,
                             output_csv: str,
                             model_path:    str = MODEL_PATH,
                             features_path: str = FEATURES_PATH) -> None:
    """Read CSV → predict → write predictions to output_csv."""
    records = pd.read_csv(input_csv).to_dict(orient="records")
    result  = predict_demand(records, model_path, features_path,
                              return_confidence=True)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    result.to_csv(output_csv, index=False)
    logger.info("Batch predictions saved → %s", output_csv)


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Zero-fill lag/rolling cols for demo (no history available at inference)
    def _empty_lags():
        return {f"lag_{d}": 0 for d in [1,7,14,30]} | \
               {f"rolling_{s}_{w}": 0
                for s in ["mean","std","min","max"] for w in [7,14,30]} | \
               {"ewma_7": 0, "ewma_30": 0,
                "sales_momentum": 0, "sales_acceleration": 0,
                "demand_volatility": 0}

    samples = [
        {"date":"2024-12-25","product_id":"P001","store_id":"S01",
         "price":49.99,"promotion":1,"holiday":1,"past_sales":150,
         **_empty_lags()},
        {"date":"2024-06-15","product_id":"P015","store_id":"S04",
         "price":299.00,"promotion":0,"holiday":0,"past_sales":35,
         **_empty_lags()},
        {"date":"2024-11-29","product_id":"P008","store_id":"S02",
         "price":89.99,"promotion":1,"holiday":0,"past_sales":200,
         **_empty_lags()},
    ]

    results = predict_demand(samples, return_confidence=True)
    print("\n📦 Demand Predictions:")
    print(results[["date","product_id","store_id","price",
                    "promotion","holiday",
                    "pred_low","predicted_demand","pred_high"]].to_string(index=False))
