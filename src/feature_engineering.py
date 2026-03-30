"""
feature_engineering.py
======================
Advanced feature engineering:
  - Lag features (1, 7, 14, 30)
  - Rolling statistics (mean, std, min, max)
  - Exponential weighted moving average (EWMA)
  - Sales momentum & acceleration
  - Promotion impact score
  - Holiday interaction features
  - Price elasticity signal
  - Demand volatility index
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lag Features
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     target_col:  str  = "demand",
                     group_cols:  list = None,
                     lags:        list = None) -> pd.DataFrame:
    """
    Add lag-n values of target_col grouped by group_cols.
    Lags: 1, 7, 14, 30 days — captures short & long-term patterns.
    """
    df    = df.copy()
    lags  = lags       or [1, 7, 14, 30]
    group = group_cols or []

    for lag in lags:
        col = f"lag_{lag}"
        df[col] = (df.groupby(group)[target_col].shift(lag)
                   if group else df[target_col].shift(lag))
        logger.info("Created %s", col)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling Statistics
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame,
                          target_col:  str  = "demand",
                          group_cols:  list = None,
                          windows:     list = None) -> pd.DataFrame:
    """
    Rolling mean, std, min, max — provides trend and volatility signals.
    Uses shift(1) to prevent data leakage.
    """
    df      = df.copy()
    windows = windows    or [7, 14, 30]
    group   = group_cols or []

    for w in windows:
        for stat in ["mean", "std", "min", "max"]:
            col = f"rolling_{stat}_{w}"
            fn  = lambda x, s=stat, w=w: getattr(x.shift(1).rolling(w, min_periods=1), s)()
            df[col] = (df.groupby(group)[target_col].transform(fn)
                       if group else fn(df[target_col]))
            logger.info("Created %s", col)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Exponential Weighted Moving Average
# ─────────────────────────────────────────────────────────────────────────────

def add_ewma_features(df: pd.DataFrame,
                       target_col: str  = "demand",
                       group_cols: list = None,
                       spans:      list = None) -> pd.DataFrame:
    """
    EWMA gives more weight to recent demand — excellent for trending products.
    """
    df    = df.copy()
    spans = spans       or [7, 30]
    group = group_cols  or []

    for span in spans:
        col = f"ewma_{span}"
        fn  = lambda x, s=span: x.shift(1).ewm(span=s, min_periods=1).mean()
        df[col] = (df.groupby(group)[target_col].transform(fn)
                   if group else fn(df[target_col]))
        logger.info("Created %s", col)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Momentum & Acceleration
# ─────────────────────────────────────────────────────────────────────────────

def add_momentum_features(df: pd.DataFrame,
                            target_col: str = "demand") -> pd.DataFrame:
    """
    Momentum  = lag_1 − lag_2  (rate of change)
    Acceleration = lag_1 - 2*lag_2 + lag_3  (change in rate of change)
    """
    df = df.copy()
    t  = df[target_col]
    df["sales_momentum"]     = t.shift(1) - t.shift(2)
    df["sales_acceleration"] = t.shift(1) - 2 * t.shift(2) + t.shift(3)
    logger.info("Created sales_momentum, sales_acceleration")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Promotion & Holiday Features
# ─────────────────────────────────────────────────────────────────────────────

def add_promotion_features(df: pd.DataFrame,
                             promo_col: str = "promotion",
                             price_col: str = "price") -> pd.DataFrame:
    """
    promo_impact   = promotion × (1 / price)    — cheap promo = highest lift
    promo_discount = promotion × price           — expensive promo absolute value
    """
    df = df.copy()
    if promo_col in df.columns and price_col in df.columns:
        df["promo_impact"]   = df[promo_col] * (1 / df[price_col].replace(0, np.nan)).fillna(0)
        df["promo_discount"] = df[promo_col] * df[price_col]
        logger.info("Created promo_impact, promo_discount")
    return df

def add_holiday_features(df: pd.DataFrame,
                          holiday_col: str = "holiday",
                          promo_col:   str = "promotion") -> pd.DataFrame:
    """Holiday × Promo interaction — double-event demand spike."""
    df = df.copy()
    if holiday_col in df.columns and promo_col in df.columns:
        df["holiday_promo"] = df[holiday_col] * df[promo_col]
        logger.info("Created holiday_promo")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Price Features
# ─────────────────────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame,
                        price_col: str = "price",
                        n_bins:    int = 5) -> pd.DataFrame:
    """
    price_bucket   = quantile bin (0–4) — non-linear price band signal
    price_log      = log(price)          — reduces skew for linear models
    price_per_unit = price / past_sales  — cost efficiency signal
    """
    df = df.copy()
    if price_col in df.columns:
        df["price_log"]    = np.log1p(df[price_col])
        df["price_bucket"] = pd.qcut(df[price_col], q=n_bins,
                                      labels=False, duplicates="drop").fillna(0)
        if "past_sales" in df.columns:
            df["price_per_unit"] = df[price_col] / (df["past_sales"].replace(0, 1))
        logger.info("Created price_log, price_bucket, price_per_unit")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Demand Volatility Index
# ─────────────────────────────────────────────────────────────────────────────

def add_volatility_index(df: pd.DataFrame,
                          target_col: str  = "demand",
                          group_cols: list = None,
                          window:     int  = 14) -> pd.DataFrame:
    """
    Coefficient of Variation (std/mean) over a rolling window.
    High volatility products need different stocking strategies.
    """
    df    = df.copy()
    group = group_cols or []
    fn    = lambda x: (x.shift(1).rolling(window, min_periods=1).std() /
                        x.shift(1).rolling(window, min_periods=1).mean().replace(0, np.nan))
    df["demand_volatility"] = (df.groupby(group)[target_col].transform(fn)
                                if group else fn(df[target_col]))
    df["demand_volatility"].fillna(0, inplace=True)
    logger.info("Created demand_volatility")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full Feature Engineering Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame,
                              target_col: str  = "demand",
                              group_cols: list = None) -> pd.DataFrame:
    """
    Applies all feature engineering steps in order and drops NaN rows
    introduced by lagging.

    Returns feature-enriched DataFrame ready for model training.
    """
    group = group_cols or []

    df = add_lag_features(df, target_col, group, lags=[1, 7, 14, 30])
    df = add_rolling_features(df, target_col, group, windows=[7, 14, 30])
    df = add_ewma_features(df, target_col, group, spans=[7, 30])
    df = add_momentum_features(df, target_col)
    df = add_promotion_features(df)
    df = add_holiday_features(df)
    df = add_price_features(df)
    df = add_volatility_index(df, target_col, group)

    before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Dropped %d NaN rows after feature engineering", before - len(df))
    logger.info("Final shape: %s | %d features", df.shape, df.shape[1] - 1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from data_preprocessing import run_preprocessing_pipeline, generate_synthetic_dataset

    RAW  = "data/raw/sales_data.csv"
    PROC = "data/processed/sales_processed.csv"
    FEAT = "data/processed/sales_featured.csv"

    if not os.path.exists(RAW):
        os.makedirs("data/raw", exist_ok=True)
        generate_synthetic_dataset().to_csv(RAW, index=False)

    df_proc = run_preprocessing_pipeline(RAW, PROC)
    df_feat = run_feature_engineering(df_proc, "demand",
                                       ["product_id", "store_id"])
    df_feat.to_csv(FEAT, index=False)
    print(f"\n✅ Feature engineering done — {df_feat.shape[1]-1} features | {len(df_feat)} rows")
