"""
evaluate_model.py
=================
Computes RMSE, MAE, R², MAPE for all trained models.
Generates 5 evaluation plots: comparison bar, actual vs predicted,
residuals, feature importance, learning curve.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve

sns.set_theme(style="whitegrid", palette="husl")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLORS = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0","#00BCD4"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error — ignores zero-demand rows."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "RMSE": round(rmse, 3),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 3),
        "R²":   round(r2_score(y_true, y_pred), 4),
        "MAPE": round(mape(np.array(y_true), np.array(y_pred)), 2),
    }

def evaluate_all_models(models: dict,
                         X_test: pd.DataFrame,
                         y_test: pd.Series) -> pd.DataFrame:
    records = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        m       = compute_metrics(y_test.values, y_pred)
        m["Model"] = name
        records.append(m)
        logger.info("%s | RMSE=%.3f  MAE=%.3f  R²=%.4f  MAPE=%.2f%%",
                    name, m["RMSE"], m["MAE"], m["R²"], m["MAPE"])

    df = (pd.DataFrame(records)[["Model","RMSE","MAE","R²","MAPE"]]
            .sort_values("RMSE").reset_index(drop=True))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(df_results: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors    = COLORS[:len(df_results)]

    # RMSE bar
    bars = axes[0].bar(df_results["Model"], df_results["RMSE"],
                        color=colors, edgecolor="black", linewidth=0.7)
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0].set_title("RMSE Comparison", fontweight="bold")
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[0].tick_params(axis="x", rotation=25)

    # R² bar
    bars2 = axes[1].bar(df_results["Model"], df_results["R²"],
                         color=colors, edgecolor="black", linewidth=0.7)
    axes[1].bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    axes[1].set_title("R² Score Comparison", fontweight="bold")
    axes[1].set_ylabel("R² (higher is better)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(0, 1.1)

    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(model, X_test, y_test,
                               model_name="Model", save_path=None):
    y_pred = model.predict(X_test)
    m      = compute_metrics(y_test.values, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=14, color="#2196F3", edgecolors="none")
    lim = [min(y_test.min(), y_pred.min()) * 0.95,
           max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1.8, label="Perfect fit")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual Demand"); ax.set_ylabel("Predicted Demand")
    ax.set_title(f"{model_name} — Actual vs Predicted", fontweight="bold")
    ax.legend()
    ax.text(0.04, 0.92,
            f"RMSE={m['RMSE']}  MAE={m['MAE']}  R²={m['R²']}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residuals(model, X_test, y_test,
                    model_name="Model", save_path=None):
    y_pred    = model.predict(X_test)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(residuals, bins=45, kde=True, color="coral",
                 edgecolor="white", ax=axes[0])
    axes[0].axvline(0, color="black", ls="--", lw=1.5)
    axes[0].set_title("Residual Distribution"); axes[0].set_xlabel("Residual")

    axes[1].scatter(y_pred, residuals, alpha=0.3, s=14,
                    color="#9C27B0", edgecolors="none")
    axes[1].axhline(0, color="red", ls="--", lw=1.5)
    axes[1].set_title("Residuals vs Fitted Values")
    axes[1].set_xlabel("Fitted Values"); axes[1].set_ylabel("Residuals")

    plt.suptitle(f"{model_name} — Residual Analysis", fontweight="bold")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(model, feature_names, top_n=25,
                              model_name="Model", save_path=None):
    estimator = model.named_steps.get("model", model)
    if not hasattr(estimator, "feature_importances_"):
        logger.warning("%s has no feature_importances_ — skipping", model_name)
        return None

    fi = (pd.Series(estimator.feature_importances_, index=feature_names)
            .sort_values(ascending=False).head(top_n))

    fig, ax = plt.subplots(figsize=(10, top_n * 0.32 + 1.5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi)))[::-1]
    fi[::-1].plot(kind="barh", ax=ax, color=colors, edgecolor="grey", linewidth=0.5)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_heatmap(df_results: pd.DataFrame, save_path: str = None):
    """Heatmap of all metrics across all models — easy visual comparison."""
    df_plot = df_results.set_index("Model")[["RMSE","MAE","MAPE","R²"]].copy()
    # Normalise each column 0-1 for colour scale (invert RMSE/MAE/MAPE)
    df_norm = df_plot.copy()
    for col in ["RMSE","MAE","MAPE"]:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = 1 - (df_norm[col] - mn) / (mx - mn + 1e-9)
    mn, mx = df_norm["R²"].min(), df_norm["R²"].max()
    df_norm["R²"] = (df_norm["R²"] - mn) / (mx - mn + 1e-9)

    fig, ax = plt.subplots(figsize=(8, len(df_results) * 0.6 + 1.5))
    sns.heatmap(df_norm, annot=df_plot.values, fmt=".3f",
                cmap="RdYlGn", linewidths=0.5, ax=ax,
                annot_kws={"size": 10})
    ax.set_title("Model Metrics Heatmap (greener = better)", fontweight="bold")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_pipeline(model_dir:     str = "models/",
                              featured_path: str = "data/processed/sales_featured.csv",
                              plots_dir:     str = "models/plots/",
                              target_col:    str = "demand") -> pd.DataFrame:
    os.makedirs(plots_dir, exist_ok=True)

    df           = pd.read_csv(featured_path)
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))
    X            = df[feature_cols]
    y            = df[target_col]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load all saved models
    skip     = {"best_model.joblib", "feature_cols.joblib",
                "best_model_name.joblib", "cv_scores.csv"}
    models   = {}
    for fname in os.listdir(model_dir):
        if fname.endswith(".joblib") and fname not in skip:
            name          = fname.replace(".joblib", "")
            models[name]  = joblib.load(os.path.join(model_dir, fname))

    df_results = evaluate_all_models(models, X_test, y_test)

    print("\n" + "═" * 60)
    print("         MODEL EVALUATION RESULTS — TEST SET")
    print("═" * 60)
    print(df_results.to_string(index=False))
    print("═" * 60)

    # Save metrics CSV
    df_results.to_csv(os.path.join(model_dir, "metrics.csv"), index=False)

    # Generate all plots
    best_name  = df_results.iloc[0]["Model"]
    best_model = models[best_name]

    plot_model_comparison(df_results,
        os.path.join(plots_dir, "model_comparison.png"))
    plot_actual_vs_predicted(best_model, X_test, y_test, best_name,
        os.path.join(plots_dir, "actual_vs_predicted.png"))
    plot_residuals(best_model, X_test, y_test, best_name,
        os.path.join(plots_dir, "residuals.png"))
    plot_feature_importance(best_model, feature_cols, model_name=best_name,
        save_path=os.path.join(plots_dir, "feature_importance.png"))
    plot_metrics_heatmap(df_results,
        os.path.join(plots_dir, "metrics_heatmap.png"))

    logger.info("All evaluation plots saved to %s", plots_dir)
    return df_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation_pipeline()
