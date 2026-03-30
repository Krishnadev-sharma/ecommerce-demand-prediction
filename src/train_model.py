"""
train_model.py
==============
Trains 6 models with GridSearchCV + cross-validation:
  1. Linear Regression       (baseline)
  2. Ridge Regression        (regularised linear)
  3. Random Forest           (ensemble bagging)
  4. Gradient Boosting       (sklearn GBM)
  5. XGBoost                 (optimised GBM)
  6. LightGBM                (fast GBM — best accuracy)

Selects best model by CV RMSE and saves all models + feature list.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

try:
    from xgboost  import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from lightgbm import LGBMRegressor
    LGB_OK = True
except ImportError:
    LGB_OK = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame,
               target_col:    str   = "demand",
               test_size:     float = 0.2,
               random_state:  int   = 42):
    X = df[[c for c in df.columns if c != target_col]]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Registry
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return sklearn Pipelines (Scaler + Estimator) for all models."""
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(n_estimators=150,
                                              random_state=42, n_jobs=-1))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(n_estimators=200,
                                                   learning_rate=0.05,
                                                   random_state=42))
        ]),
    }
    if XGB_OK:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(n_estimators=300, learning_rate=0.05,
                                     max_depth=6, subsample=0.8,
                                     colsample_bytree=0.8,
                                     random_state=42, verbosity=0))
        ])
    if LGB_OK:
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                      num_leaves=63, subsample=0.8,
                                      colsample_bytree=0.8,
                                      random_state=42, verbose=-1))
        ])
    logger.info("Models registered: %s", list(models.keys()))
    return models


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameter Grids
# ─────────────────────────────────────────────────────────────────────────────

def get_param_grids() -> dict:
    """GridSearchCV parameter grids — pipeline keys use 'model__' prefix."""
    grids = {
        "LinearRegression": {},
        "Ridge": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "RandomForest": {
            "model__n_estimators":     [100, 200],
            "model__max_depth":        [None, 10, 20],
            "model__min_samples_split":[2, 5],
        },
        "GradientBoosting": {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth":     [3, 5],
        },
    }
    if XGB_OK:
        grids["XGBoost"] = {
            "model__n_estimators":  [200, 300],
            "model__learning_rate": [0.03, 0.05],
            "model__max_depth":     [4, 6],
            "model__subsample":     [0.8, 1.0],
        }
    if LGB_OK:
        grids["LightGBM"] = {
            "model__n_estimators":  [200, 300],
            "model__learning_rate": [0.03, 0.05],
            "model__num_leaves":    [31, 63],
            "model__subsample":     [0.8, 1.0],
        }
    return grids


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      cv: int = 3) -> tuple[dict, dict]:
    """
    Trains every model with GridSearchCV.
    Returns (trained_models dict, cv_rmse_scores dict).
    """
    models      = get_models()
    param_grids = get_param_grids()
    trained     = {}
    cv_scores   = {}

    for name, pipeline in models.items():
        logger.info("⏳ Training %s …", name)
        grid = param_grids.get(name, {})

        if grid:
            gs = GridSearchCV(pipeline, param_grid=grid, cv=cv,
                              scoring="neg_root_mean_squared_error",
                              n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            best      = gs.best_estimator_
            rmse_cv   = -gs.best_score_
            logger.info("%s ✓ | Best params: %s | CV RMSE: %.4f",
                        name, gs.best_params_, rmse_cv)
        else:
            pipeline.fit(X_train, y_train)
            best    = pipeline
            scores  = cross_val_score(pipeline, X_train, y_train,
                                       cv=cv, scoring="neg_root_mean_squared_error",
                                       n_jobs=-1)
            rmse_cv = -scores.mean()
            logger.info("%s ✓ | CV RMSE: %.4f", name, rmse_cv)

        trained[name]   = best
        cv_scores[name] = round(rmse_cv, 4)

    return trained, cv_scores


# ─────────────────────────────────────────────────────────────────────────────
# 5. Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved → %s", path)

def load_model(path: str):
    model = joblib.load(path)
    logger.info("Loaded ← %s", path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(featured_path: str = "data/processed/sales_featured.csv",
                           model_dir:     str = "models/",
                           target_col:    str = "demand") -> dict:
    """
    Complete training pipeline:
    load → split → train → save best model + all models.
    Returns results dict with models, scores, and test data.
    """
    df = pd.read_csv(featured_path)
    logger.info("Loaded featured data: %s", df.shape)

    X_train, X_test, y_train, y_test = split_data(df, target_col)
    trained, cv_scores = train_all_models(X_train, y_train)

    best_name  = min(cv_scores, key=cv_scores.get)
    best_model = trained[best_name]
    logger.info("🏆 Best model: %s  (CV RMSE=%.4f)", best_name, cv_scores[best_name])

    # Save all models
    for name, mdl in trained.items():
        save_model(mdl, os.path.join(model_dir, f"{name}.joblib"))

    # Save best model & metadata
    save_model(best_model, os.path.join(model_dir, "best_model.joblib"))
    feature_cols = [c for c in df.columns if c != target_col]
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.joblib"))
    joblib.dump(best_name,    os.path.join(model_dir, "best_model_name.joblib"))

    # Save CV scores CSV for dashboard
    pd.DataFrame(
        [{"model": k, "cv_rmse": v} for k, v in cv_scores.items()]
    ).to_csv(os.path.join(model_dir, "cv_scores.csv"), index=False)

    print("\n" + "═" * 50)
    print("       CROSS-VALIDATION RMSE SCORES")
    print("═" * 50)
    for name, score in sorted(cv_scores.items(), key=lambda x: x[1]):
        marker = " 🏆" if name == best_name else ""
        print(f"  {name:<22} {score:.4f}{marker}")
    print("═" * 50)

    return {"trained_models": trained, "cv_scores": cv_scores,
            "best_model_name": best_name,
            "X_test": X_test, "y_test": y_test}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_training_pipeline()
