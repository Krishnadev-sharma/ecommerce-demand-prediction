"""
data_preprocessing.py
=====================
Advanced data ingestion, cleaning, and preprocessing pipeline.
Handles missing values, encoding, scaling, and date decomposition.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic Dataset Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_records: int = 15_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic e-commerce sales dataset.
    Includes seasonality, price elasticity, promo lifts & holiday spikes.
    """
    rng = np.random.default_rng(seed)
    dates       = pd.date_range("2021-01-01", periods=365 * 3, freq="D")
    product_ids = [f"P{str(i).zfill(3)}" for i in range(1, 31)]   # 30 products
    store_ids   = [f"S{str(i).zfill(2)}" for i in range(1, 8)]    # 7 stores
    categories  = {"Electronics": 0.4, "Clothing": 0.3, "Food": 0.3}

    cat_list  = list(categories.keys())
    cat_probs = list(categories.values())

    rows = []
    for _ in range(n_records):
        date      = rng.choice(dates)
        product   = rng.choice(product_ids)
        store     = rng.choice(store_ids)
        category  = rng.choice(cat_list, p=cat_probs)
        price     = round(rng.uniform(5.0, 800.0), 2)
        promotion = int(rng.random() < 0.25)
        holiday   = int(pd.Timestamp(date).month in [11, 12] and pd.Timestamp(date).day >= 20)

        # Seasonality boost
        date_ts = pd.Timestamp(date); season_boost = 20 if date_ts.month in [11, 12] else (10 if date_ts.month in [6, 7] else 0)

        # Demand model
        base        = rng.integers(10, 180)
        promo_boost = promotion * rng.integers(30, 100)
        hol_boost   = holiday   * rng.integers(15, 60)
        price_eff   = max(0, int((800 - price) / 25))
        noise       = rng.integers(-15, 15)
        demand      = max(1, base + promo_boost + hol_boost + price_eff + season_boost + noise)
        past_sales  = max(0, demand + rng.integers(-40, 40))

        date_ts2=pd.Timestamp(date)
        rows.append([str(date_ts2.date()), product, store, category,
                     price, promotion, holiday, past_sales, demand])

    df = pd.DataFrame(rows, columns=[
        "date", "product_id", "store_id", "category",
        "price", "promotion", "holiday", "past_sales", "demand"
    ])
    logger.info("Synthetic dataset generated — %d rows", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["date"])
    logger.info("Loaded %s — shape %s", filepath, df.shape)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation for numerics, mode for categoricals."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    logger.info("Missing values handled")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows", before - len(df))
    return df

def remove_outliers_iqr(df: pd.DataFrame, col: str = "demand") -> pd.DataFrame:
    """Remove extreme outliers using 1.5×IQR rule."""
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    before = len(df)
    df     = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    logger.info("Outlier removal on '%s': dropped %d rows", col, before - len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Decompose datetime into rich calendar features."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    df["year"]         = df[date_col].dt.year
    df["month"]        = df[date_col].dt.month
    df["day"]          = df[date_col].dt.day
    df["day_of_week"]  = df[date_col].dt.dayofweek
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"]      = df[date_col].dt.quarter
    df["is_weekend"]   = (df[date_col].dt.dayofweek >= 5).astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    df["is_month_start"]= df[date_col].dt.is_month_start.astype(int)

    # Cyclical encoding for month & day_of_week (helps linear models)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)
    logger.info("Date features extracted from '%s'", date_col)
    return df

def encode_categoricals(df: pd.DataFrame,
                         label_cols: list = None,
                         ohe_cols:   list = None) -> pd.DataFrame:
    df         = df.copy()
    label_cols = label_cols or []
    ohe_cols   = ohe_cols   or []
    le         = LabelEncoder()
    for col in label_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    if ohe_cols:
        existing = [c for c in ohe_cols if c in df.columns]
        df = pd.get_dummies(df, columns=existing, drop_first=True)
    logger.info("Encoding done — label: %s | OHE: %s", label_cols, ohe_cols)
    return df

def scale_features(df: pd.DataFrame,
                   feature_cols: list,
                   scaler: StandardScaler = None):
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(input_path:  str,
                                output_path: str,
                                save: bool = True) -> pd.DataFrame:
    """
    load → clean → outlier removal → date features → encode → save
    """
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers_iqr(df, col="demand")
    df = extract_date_features(df, "date")
    df = encode_categoricals(df,
                              label_cols=["product_id", "store_id"],
                              ohe_cols=["category"])
    df.drop(columns=["date"], inplace=True, errors="ignore")

    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Processed data saved → %s", output_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RAW  = "data/raw/sales_data.csv"
    PROC = "data/processed/sales_processed.csv"

    if not os.path.exists(RAW):
        os.makedirs("data/raw", exist_ok=True)
        generate_synthetic_dataset().to_csv(RAW, index=False)
        logger.info("Raw data saved → %s", RAW)

    df = run_preprocessing_pipeline(RAW, PROC)
    print(f"\n✅ Preprocessing done — shape: {df.shape}")
    print(df.head())
