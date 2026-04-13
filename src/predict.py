"""
predict.py - CRASH-PROOF v6
=============================
Inference pipeline with complete error handling.
Never crashes on numpy/sklearn version mismatch.
Falls back to heuristic automatically.
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH    = "models/best_model.joblib"
FEATURES_PATH = "models/feature_cols.joblib"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Heuristic Fallback (always works — no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_predict(price:      float,
                       promotion:  int,
                       holiday:    int,
                       past_sales: int,
                       stock:      int = 100) -> tuple:
    """
    Smart rule-based demand estimate.
    Used when ML model fails to load or predict.
    Returns (prediction, low_estimate, high_estimate)
    """
    base         = max(float(past_sales), 50.0)
    promo_mult   = 1.40 if promotion else 1.0
    holiday_mult = 1.25 if holiday   else 1.0
    price_factor = max(0.2, 1.0 - price / 100_000.0)
    stock_factor = min(1.0, stock / 200.0) if stock > 0 else 0.5

    pred = int(base * promo_mult * holiday_mult * price_factor * stock_factor)
    pred = max(pred, 1)
    return pred, max(int(pred * 0.82), 1), int(pred * 1.18)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features(records: list, feature_cols: list) -> pd.DataFrame:
    """
    Convert raw input dicts into model-ready DataFrame.
    Applies the same transformations as the training pipeline.
    Any missing columns are filled with 0.
    """
    df = pd.DataFrame(records)

    # ── Date decomposition ──
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"]           = df["date"].dt.year
        df["month"]          = df["date"].dt.month
        df["day"]            = df["date"].dt.day
        df["day_of_week"]    = df["date"].dt.dayofweek
        df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"]        = df["date"].dt.quarter
        df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
        df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"]        = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]        = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df.drop(columns=["date"], inplace=True, errors="ignore")

    # ── Encode IDs ──
    for col in ["product_id", "store_id"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes

    # ── Promo features ──
    if "promotion" in df.columns and "price" in df.columns:
        df["promo_impact"]   = df["promotion"] * (1.0 / df["price"].replace(0, np.nan)).fillna(0)
        df["promo_discount"] = df["promotion"] * df["price"]

    # ── Holiday interaction ──
    if "holiday" in df.columns and "promotion" in df.columns:
        df["holiday_promo"] = df["holiday"] * df["promotion"]

    # ── Price features ──
    if "price" in df.columns:
        df["price_log"]    = np.log1p(df["price"])
        df["price_bucket"] = pd.cut(df["price"],
                                     bins=[0, 50, 100, 200, 350, np.inf],
                                     labels=False).fillna(0)
        if "past_sales" in df.columns:
            df["price_per_unit"] = df["price"] / (df["past_sales"].replace(0, 1))

    # ── Fill any missing feature columns with 0 ──
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Safe Model Loader
# ─────────────────────────────────────────────────────────────────────────────

def safe_load_model(model_path: str, features_path: str):
    """
    Load model and feature columns safely.
    Returns (model, feature_cols) or (None, None) on any error.
    """
    try:
        import joblib
        model        = joblib.load(model_path)
        feature_cols = joblib.load(features_path)
        logger.info("Model loaded successfully from %s", model_path)
        return model, feature_cols
    except Exception as e:
        logger.warning("Model load failed (%s) — will use heuristic", str(e)[:80])
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Prediction Function (CRASH-PROOF)
# ─────────────────────────────────────────────────────────────────────────────

def predict_demand(records: list,
                    model_path:         str  = MODEL_PATH,
                    features_path:      str  = FEATURES_PATH,
                    return_confidence:  bool = False) -> pd.DataFrame:
    """
    Predict demand for one or more records.

    Tries ML model first → falls back to heuristic on any failure.
    NEVER raises an exception — always returns a valid DataFrame.

    Parameters
    ----------
    records           : list of dicts with product/store/date features
    model_path        : path to saved best_model.joblib
    features_path     : path to saved feature_cols.joblib
    return_confidence : if True, adds pred_low and pred_high columns

    Returns
    -------
    pd.DataFrame with original fields + predicted_demand
                 (+ pred_low, pred_high if return_confidence=True)
    """
    result_df = pd.DataFrame(records).copy()
    predictions, lows, highs = [], [], []

    # ── Try ML model ──
    model, feature_cols = safe_load_model(model_path, features_path)
    ml_available = model is not None and feature_cols is not None

    for i, record in enumerate(records):
        ml_success = False

        if ml_available:
            try:
                X    = build_features([record], feature_cols)
                pred = float(model.predict(X)[0])
                pred = max(int(round(pred)), 1)
                low  = max(int(pred * 0.82), 1)
                high = int(pred * 1.18)
                ml_success = True
            except Exception as e:
                logger.warning("ML prediction failed for record %d: %s", i, str(e)[:60])

        if not ml_success:
            # Heuristic fallback for this record
            pred, low, high = heuristic_predict(
                price      = float(record.get("price",      999)),
                promotion  = int(record.get("promotion",    0)),
                holiday    = int(record.get("holiday",      0)),
                past_sales = int(record.get("past_sales",   100)),
                stock      = int(record.get("stock_level",  100)),
            )

        predictions.append(pred)
        lows.append(low)
        highs.append(high)

    result_df["predicted_demand"] = predictions
    if return_confidence:
        result_df["pred_low"]  = lows
        result_df["pred_high"] = highs

    logger.info("Predicted %d records (ML=%s)", len(records), ml_available)
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Batch CSV Prediction
# ─────────────────────────────────────────────────────────────────────────────

def batch_predict_from_csv(input_csv:  str,
                             output_csv: str,
                             model_path:    str = MODEL_PATH,
                             features_path: str = FEATURES_PATH) -> None:
    """Read CSV → predict → write to output_csv."""
    df      = pd.read_csv(input_csv)
    records = df.to_dict(orient="records")
    result  = predict_demand(records, model_path, features_path,
                              return_confidence=True)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    result.to_csv(output_csv, index=False)
    logger.info("Batch predictions saved → %s", output_csv)


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def empty_lags():
        return ({f"lag_{d}": 0 for d in [1, 7, 14, 30]} |
                {f"rolling_{s}_{w}": 0
                 for s in ["mean","std","min","max"] for w in [7,14,30]} |
                {"ewma_7":0,"ewma_30":0,
                 "sales_momentum":0,"sales_acceleration":0,
                 "demand_volatility":0})

    samples = [
        {"date":"2024-12-25","product_id":"P001","store_id":"S01",
         "price":49.99,"promotion":1,"holiday":1,
         "past_sales":150,"stock_level":200, **empty_lags()},
        {"date":"2024-06-15","product_id":"P015","store_id":"S04",
         "price":299.00,"promotion":0,"holiday":0,
         "past_sales":35,"stock_level":80, **empty_lags()},
        {"date":"2024-11-29","product_id":"P008","store_id":"S02",
         "price":89.99,"promotion":1,"holiday":0,
         "past_sales":200,"stock_level":500, **empty_lags()},
    ]

    results = predict_demand(samples, return_confidence=True)
    print("\n📦 Demand Predictions:")
    print(results[["date","product_id","store_id","price",
                    "promotion","holiday",
                    "pred_low","predicted_demand","pred_high"]].to_string(index=False))
