"""
streamlit_app.py
================
Professional E-commerce Demand Prediction Dashboard
  🏠 Home          — project overview & quick stats
  📊 EDA           — upload CSV, explore 8 interactive charts
  🤖 Predict       — single-record form + batch CSV upload
  📈 Performance   — model metrics table + 5 evaluation plots
  📚 Documentation — feature guide & run instructions

Run: streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Predictor Pro",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem 1.5rem; border-radius: 12px; color: white;
    text-align: center; margin-bottom: 0.5rem;
  }
  .metric-card h2 { margin: 0; font-size: 2rem; }
  .metric-card p  { margin: 0; font-size: 0.9rem; opacity: 0.85; }
  .green-card { background: linear-gradient(135deg,#11998e,#38ef7d); }
  .orange-card{ background: linear-gradient(135deg,#f7971e,#ffd200); color:#333; }
  .red-card   { background: linear-gradient(135deg,#cb2d3e,#ef473a); }
  .section-title { font-size:1.4rem; font-weight:700;
                   border-left:4px solid #667eea; padding-left:10px;
                   margin:1.5rem 0 0.8rem 0; }
  div[data-testid="metric-container"] {
    border: 1px solid #e0e0e0; border-radius: 10px;
    padding: 0.6rem 1rem; background:#fafafa;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts(model_dir="models/"):
    models, feature_cols, best_name = {}, None, None
    if not os.path.isdir(model_dir):
        return models, feature_cols, best_name
    skip = {"feature_cols.joblib","best_model_name.joblib","best_model.joblib"}
    for f in os.listdir(model_dir):
        if f.endswith(".joblib") and f not in skip:
            name = f.replace(".joblib","")
            try: models[name] = joblib.load(os.path.join(model_dir, f))
            except: pass
    fc = os.path.join(model_dir,"feature_cols.joblib")
    bn = os.path.join(model_dir,"best_model_name.joblib")
    if os.path.exists(fc): feature_cols = joblib.load(fc)
    if os.path.exists(bn): best_name    = joblib.load(bn)
    return models, feature_cols, best_name


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def metric_card(title, value, color=""):
    st.markdown(
        f'<div class="metric-card {color}"><h2>{value}</h2><p>{title}</p></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Demand Predictor")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Home",
        "📊 EDA & Analysis",
        "🤖 Predict Demand",
        "📈 Model Performance",
        "📚 Documentation",
    ])
    st.markdown("---")
    models, feature_cols, best_name = load_artifacts()
    if models:
        st.success(f"✅ {len(models)} model(s) loaded")
        if best_name:
            st.info(f"🏆 Best: **{best_name}**")
    else:
        st.warning("⚠️ No trained models found.\nRun `python src/train_model.py`")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📦 E-commerce Product Demand Prediction")
    st.markdown("**AI-powered demand forecasting for smarter inventory decisions.**")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("ML Models", str(len(models)) if models else "0", "")
    with c2: metric_card("Features Engineered", "35+", "green-card")
    with c3: metric_card("Best Model", best_name or "N/A", "orange-card")
    with c4: metric_card("Metrics Tracked", "RMSE · MAE · R² · MAPE", "red-card")

    st.markdown("---")
    st.markdown('<div class="section-title">🗺️ How It Works</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Pipeline Steps**
        1. 🧹 **Preprocess** — clean, encode, extract date features
        2. ⚙️ **Feature Engineering** — lags, rolling stats, EWMA, interactions
        3. 🤖 **Train** — 6 models with GridSearchCV
        4. 📊 **Evaluate** — RMSE, MAE, R², MAPE on hold-out test set
        5. 🔮 **Predict** — single or batch inference with confidence intervals
        """)
    with col2:
        st.markdown("""
        **Models Trained**
        | Model | Type |
        |-------|------|
        | Linear Regression | Baseline |
        | Ridge Regression | Regularised |
        | Random Forest | Ensemble |
        | Gradient Boosting | Sequential |
        | XGBoost | Optimised GBM |
        | LightGBM | Fast GBM ⭐ |
        """)

    st.markdown("---")
    st.info("👈 Use the sidebar to navigate. Start with **EDA** after uploading your data.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Analysis":
    st.title("📊 Exploratory Data Analysis")

    uploaded = st.file_uploader("Upload your sales CSV", type=["csv"])
    if not uploaded:
        # Try default dataset
        default = "data/raw/sales_data.csv"
        if os.path.exists(default):
            st.info("📂 Showing default synthetic dataset. Upload your own to replace it.")
            df = prep_df(pd.read_csv(default))
        else:
            st.info("👆 Upload a CSV file to begin EDA.")
            st.stop()
    else:
        df = prep_df(pd.read_csv(uploaded))

    # Summary row
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",        f"{len(df):,}")
    c2.metric("Columns",     len(df.columns))
    c3.metric("Date Range",  f"{df['date'].min().date() if 'date' in df.columns else 'N/A'}")
    c4.metric("Avg Demand",  f"{df['demand'].mean():.1f}" if "demand" in df.columns else "N/A")

    with st.expander("📋 Data Preview & Info", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Data Types**")
            st.dataframe(df.dtypes.rename("dtype"), use_container_width=True)
        with col_b:
            st.write("**Missing Values**")
            mv = df.isnull().sum()
            st.dataframe(mv[mv > 0].rename("missing") if mv.any() else
                         pd.Series({"All columns": "No missing values"}),
                         use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📈 Charts</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # 1. Sales Trend
    with col1:
        st.subheader("Sales Trend Over Time")
        if "date" in df.columns and "demand" in df.columns:
            freq = st.selectbox("Resample frequency", ["D","W","M"], index=1, key="freq")
            trend = df.set_index("date")["demand"].resample(freq).sum().reset_index()
            fig, ax = plt.subplots(figsize=(7,3.5))
            ax.plot(trend["date"], trend["demand"], lw=1.5, color="#2196F3")
            ax.fill_between(trend["date"], trend["demand"], alpha=0.12, color="#2196F3")
            ax.set_title("Demand Trend"); ax.set_xlabel(""); ax.set_ylabel("Demand")
            plt.tight_layout(); st.pyplot(fig)

    # 2. Demand Distribution
    with col2:
        st.subheader("Demand Distribution")
        if "demand" in df.columns:
            fig, ax = plt.subplots(figsize=(7,3.5))
            sns.histplot(df["demand"], bins=40, kde=True, color="#E91E63", ax=ax)
            ax.axvline(df["demand"].mean(), color="navy", ls="--", label="Mean")
            ax.axvline(df["demand"].median(), color="orange", ls="--", label="Median")
            ax.legend(); ax.set_title("Demand Distribution")
            plt.tight_layout(); st.pyplot(fig)

    col3, col4 = st.columns(2)

    # 3. Correlation Heatmap
    with col3:
        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=[np.number])
        if len(num_df.columns) >= 2:
            fig, ax = plt.subplots(figsize=(7,5))
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                        linewidths=0.5, ax=ax, annot_kws={"size":8})
            ax.set_title("Feature Correlations")
            plt.tight_layout(); st.pyplot(fig)

    # 4. Promo vs Demand
    with col4:
        st.subheader("Promotion vs Demand")
        if "promotion" in df.columns and "demand" in df.columns:
            fig, ax = plt.subplots(figsize=(7,5))
            sns.boxplot(data=df, x="promotion", y="demand", ax=ax,
                        palette=["#4C72B0","#DD8452"])
            ax.set_xticklabels(["No Promo","Promo"])
            ax.set_title("Demand Distribution by Promotion")
            plt.tight_layout(); st.pyplot(fig)

    col5, col6 = st.columns(2)

    # 5. Monthly pattern
    with col5:
        st.subheader("Monthly Demand Pattern")
        if "date" in df.columns and "demand" in df.columns:
            df["month"] = df["date"].dt.month
            monthly = df.groupby("month")["demand"].mean()
            fig, ax = plt.subplots(figsize=(7,3.8))
            bars = ax.bar(monthly.index, monthly.values,
                          color=plt.cm.viridis(np.linspace(0.2,0.8,12)),
                          edgecolor="black", lw=0.5)
            ax.set_xticks(range(1,13))
            ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                                  "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)
            ax.set_title("Avg Demand by Month"); ax.set_ylabel("Avg Demand")
            plt.tight_layout(); st.pyplot(fig)

    # 6. Top Products
    with col6:
        st.subheader("Top 10 Products by Demand")
        if "product_id" in df.columns and "demand" in df.columns:
            top = df.groupby("product_id")["demand"].sum().nlargest(10)
            fig, ax = plt.subplots(figsize=(7,3.8))
            top[::-1].plot(kind="barh", ax=ax, color="#4CAF50", edgecolor="black", lw=0.5)
            ax.set_title("Top 10 Products"); ax.set_xlabel("Total Demand")
            plt.tight_layout(); st.pyplot(fig)

    col7, col8 = st.columns(2)

    # 7. Price vs Demand
    with col7:
        st.subheader("Price vs Demand")
        if "price" in df.columns and "demand" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            c_col = df["promotion"] if "promotion" in df.columns else "steelblue"
            sc = ax.scatter(df["price"], df["demand"], c=c_col,
                            cmap="RdYlGn", alpha=0.35, s=12)
            if "promotion" in df.columns:
                plt.colorbar(sc, ax=ax, label="Promotion")
            ax.set_xlabel("Price"); ax.set_ylabel("Demand")
            ax.set_title("Price vs Demand")
            plt.tight_layout(); st.pyplot(fig)

    # 8. Holiday Impact
    with col8:
        st.subheader("Holiday vs Non-Holiday Demand")
        if "holiday" in df.columns and "demand" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            hol = df.groupby("holiday")["demand"].mean()
            ax.bar(["Non-Holiday","Holiday"], hol.values,
                   color=["#4C72B0","#C44E52"], edgecolor="black")
            for i,(v) in enumerate(hol.values):
                ax.text(i, v+1, f"{v:.1f}", ha="center", fontweight="bold")
            ax.set_title("Avg Demand: Holiday vs Non-Holiday")
            ax.set_ylabel("Average Demand")
            plt.tight_layout(); st.pyplot(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Demand":
    st.title("🤖 Demand Prediction")

    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction"])

    with tab1:
        st.markdown('<div class="section-title">Enter Product Details</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📅 Date & IDs**")
            date       = st.date_input("Date")
            product_id = st.text_input("Product ID", "P001")
            store_id   = st.text_input("Store ID",   "S01")

        with col2:
            st.markdown("**💰 Price & Flags**")
            price     = st.number_input("Price (₹)", 1.0, 10000.0, 99.99, 10.0)
            promotion = st.radio("Promotion Active?", [0,1],
                                  format_func=lambda x: "✅ Yes" if x else "❌ No",
                                  horizontal=True)
            holiday   = st.radio("Is Holiday?", [0,1],
                                  format_func=lambda x: "🎉 Yes" if x else "🗓️ No",
                                  horizontal=True)

        with col3:
            st.markdown("**📦 Historical Context**")
            past_sales = st.number_input("Past Sales", 0, 10000, 100)
            lag_1      = st.number_input("Yesterday's Demand (lag_1)", 0, 5000, 0)
            lag_7      = st.number_input("7-Day Ago Demand (lag_7)",  0, 5000, 0)

        st.markdown("---")
        if st.button("🔮 Predict Demand", type="primary", use_container_width=True):
            def zero_lags():
                return ({f"lag_{d}": 0 for d in [1,7,14,30]} |
                        {f"rolling_{s}_{w}": 0
                         for s in ["mean","std","min","max"] for w in [7,14,30]} |
                        {"ewma_7":0,"ewma_30":0,"sales_momentum":0,
                         "sales_acceleration":0,"demand_volatility":0})

            record = {"date":str(date),"product_id":product_id,
                      "store_id":store_id,"price":price,
                      "promotion":promotion,"holiday":holiday,
                      "past_sales":past_sales, **zero_lags()}
            record["lag_1"] = lag_1
            record["lag_7"] = lag_7

            if models and feature_cols:
                from predict import predict_demand
                res  = predict_demand([record], return_confidence=True)
                pred = int(res["predicted_demand"].iloc[0])
                low  = int(res["pred_low"].iloc[0])
                high = int(res["pred_high"].iloc[0])
            else:
                pred = int(past_sales*(1+0.3*promotion+0.15*holiday)*max(0.5,1-price/800))
                low  = int(pred * 0.85)
                high = int(pred * 1.15)
                st.warning("⚠️ Heuristic estimate — train models for ML predictions.")

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.metric("📉 Low Estimate",      f"{low:,} units")
            r2.metric("📦 Predicted Demand",  f"{pred:,} units", delta="Point estimate")
            r3.metric("📈 High Estimate",     f"{high:,} units")

            # Demand gauge
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.barh(["Demand Range"], [high - low], left=[low],
                    color="#4CAF50", height=0.4, alpha=0.6, label="Confidence Range")
            ax.barh(["Demand Range"], [1], left=[pred],
                    color="#E91E63", height=0.5, label=f"Prediction: {pred}")
            ax.set_xlabel("Units"); ax.legend(loc="lower right")
            ax.set_title("Demand Forecast with Confidence Interval")
            st.pyplot(fig)

    with tab2:
        st.markdown('<div class="section-title">Batch Prediction via CSV</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        Upload a CSV with the same columns as your training data.
        Required: `date, product_id, store_id, price, promotion, holiday, past_sales`
        """)
        batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")

        if batch_file:
            df_in = pd.read_csv(batch_file)
            st.write(f"📋 {len(df_in)} records loaded")
            st.dataframe(df_in.head(), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", type="primary"):
                if models and feature_cols:
                    from predict import predict_demand
                    result = predict_demand(df_in.to_dict("records"),
                                            return_confidence=True)
                    st.success(f"✅ Predicted demand for {len(result)} records!")
                    st.dataframe(result, use_container_width=True)

                    # Download
                    csv_bytes = result.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Predictions CSV", csv_bytes,
                                        "batch_predictions.csv", "text/csv",
                                        use_container_width=True)

                    # Quick chart
                    if "predicted_demand" in result.columns:
                        fig, ax = plt.subplots(figsize=(10, 3))
                        ax.bar(range(len(result)), result["predicted_demand"],
                               color="#2196F3", alpha=0.75)
                        ax.fill_between(range(len(result)),
                                         result["pred_low"], result["pred_high"],
                                         alpha=0.2, color="green", label="CI")
                        ax.set_title("Batch Predictions")
                        ax.set_xlabel("Record Index"); ax.set_ylabel("Predicted Demand")
                        ax.legend(); st.pyplot(fig)
                else:
                    st.error("Train models first: `python src/train_model.py`")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    # Metrics table
    metrics_path = "models/metrics.csv"
    if os.path.exists(metrics_path):
        df_m = pd.read_csv(metrics_path)
        st.markdown('<div class="section-title">📊 Test Set Metrics</div>',
                    unsafe_allow_html=True)

        # Highlight best
        best_row = df_m.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🏆 Best Model",  best_row["Model"])
        c2.metric("RMSE",           f"{best_row['RMSE']:.3f}")
        c3.metric("R² Score",       f"{best_row['R²']:.4f}")
        c4.metric("MAPE",           f"{best_row['MAPE']:.2f}%")

        st.dataframe(df_m.style.highlight_min(["RMSE","MAE","MAPE"], color="#c8f5c8")
                                .highlight_max(["R²"], color="#c8f5c8"),
                     use_container_width=True)

    # CV scores
    cv_path = "models/cv_scores.csv"
    if os.path.exists(cv_path):
        st.markdown('<div class="section-title">🔁 Cross-Validation RMSE</div>',
                    unsafe_allow_html=True)
        df_cv = pd.read_csv(cv_path).sort_values("cv_rmse")
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#4CAF50" if i == 0 else "#90CAF9"
                  for i in range(len(df_cv))]
        bars = ax.bar(df_cv["model"], df_cv["cv_rmse"],
                      color=colors, edgecolor="black", lw=0.7)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_title("CV RMSE — All Models", fontweight="bold")
        ax.set_ylabel("RMSE"); ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    # Evaluation plots
    plots_dir = "models/plots"
    if os.path.isdir(plots_dir):
        st.markdown('<div class="section-title">🖼️ Evaluation Plots</div>',
                    unsafe_allow_html=True)

        plot_map = {
            "Model Comparison (RMSE + R²)":   "model_comparison.png",
            "Metrics Heatmap":                 "metrics_heatmap.png",
            "Actual vs Predicted":             "actual_vs_predicted.png",
            "Residual Analysis":               "residuals.png",
            "Feature Importance":              "feature_importance.png",
        }
        for title, fname in plot_map.items():
            path = os.path.join(plots_dir, fname)
            if os.path.exists(path):
                st.subheader(title)
                st.image(path, use_column_width=True)
    else:
        st.info("Run `python src/evaluate_model.py` to generate evaluation plots.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DOCUMENTATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📚 Documentation":
    st.title("📚 Documentation")

    st.markdown("""
    ## 🚀 Quick Start

    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Run full pipeline
    python src/data_preprocessing.py
    python src/feature_engineering.py
    python src/train_model.py
    python src/evaluate_model.py

    # 3. Launch dashboard
    streamlit run app/streamlit_app.py
    ```

    ---

    ## 🧩 Feature Categories

    | Category | Features |
    |----------|----------|
    | **Calendar** | year, month, day, day_of_week, week_of_year, quarter, is_weekend, is_month_end, is_month_start |
    | **Cyclical** | month_sin, month_cos, dow_sin, dow_cos |
    | **Lag** | lag_1, lag_7, lag_14, lag_30 |
    | **Rolling** | rolling_mean/std/min/max for 7, 14, 30 days |
    | **EWMA** | ewma_7, ewma_30 |
    | **Momentum** | sales_momentum, sales_acceleration |
    | **Price** | price_log, price_bucket, price_per_unit |
    | **Promo** | promo_impact, promo_discount, holiday_promo |
    | **Volatility** | demand_volatility |

    ---

    ## 📊 Models

    | Model | Strengths |
    |-------|-----------|
    | Linear Regression | Fast baseline, interpretable |
    | Ridge | Handles multicollinearity |
    | Random Forest | Robust, handles non-linearity |
    | Gradient Boosting | High accuracy, slower training |
    | XGBoost | Fast GBM with regularisation |
    | LightGBM | Fastest, best on large datasets ⭐ |

    ---

    ## 📏 Metrics Explained

    | Metric | Formula | Ideal |
    |--------|---------|-------|
    | **RMSE** | √(mean((y-ŷ)²)) | Lower ↓ |
    | **MAE**  | mean(|y-ŷ|)     | Lower ↓ |
    | **R²**   | 1 - SS_res/SS_tot | Higher ↑ (max 1.0) |
    | **MAPE** | mean(|y-ŷ|/y)×100 | Lower ↓ |
    """)
