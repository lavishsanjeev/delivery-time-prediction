import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. PAGE SETUP & STYLING
# =========================================================
st.set_page_config(
    page_title="DeliveryIQ | Random Forest Delivery Prediction",
    page_icon="🚚",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; max-width: 1400px; }
div[data-testid="stMetric"] {
    background: #171f2e !important;
    border: 1px solid #334155 !important;
    padding: 16px 18px !important;
    border-radius: 12px !important;
}
div[data-testid="stMetricLabel"] p { color: #cbd5e1 !important; font-size: 0.9rem; }
div[data-testid="stMetricValue"] * { color: #ffffff !important; }
.hero {
    padding: 22px 26px;
    border-radius: 14px;
    background: linear-gradient(120deg, #17172d, #373166);
    color: white;
    margin-bottom: 20px;
}
.hero h1 { color: white; margin: 0 0 6px 0; font-size: 2.1rem; }
.hero p { color: #d4d1ee; margin: 0; font-size: 1rem; }
.tag {
    display: inline-block;
    padding: 4px 10px;
    margin: 3px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. HELPER FUNCTIONS: COLUMN DETECTION & CLEANING
# =========================================================
def detect_columns(df):
    """Detects target, tracking IDs, numeric features, and categorical features."""
    cols = df.columns.tolist()

    # Target: looks for delivery duration or falls back to last numeric column
    target_names = ["delivery_time_min", "delivery_min", "delivery_time", "actual_delivery_min", "duration", "target"]
    target = next((c for c in cols if c.lower().strip().replace(" ", "_") in target_names), None)
    if not target:
        target = next((c for c in cols if "delivery" in c.lower() and ("time" in c.lower() or "min" in c.lower())), cols[-1])

    # ID columns: exclude order_id, tracking_id, or strictly unique index columns
    id_names = ["order_id", "id", "order_no", "order_number", "tracking_id", "trip_id"]
    id_cols = [c for c in cols if c.lower().strip().replace(" ", "_") in id_names or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() == len(df))]

    # Features: remaining columns split by numeric vs categorical
    feature_candidates = [c for c in cols if c != target and c not in id_cols]
    numeric_cols = []
    categorical_cols = []

    for c in feature_candidates:
        num_s = pd.to_numeric(df[c], errors="coerce")
        if num_s.notnull().mean() > 0.8 and df[c].nunique() > 6:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return target, id_cols, numeric_cols, categorical_cols


def clean_dataset(df, target_col, numeric_cols, categorical_cols):
    """Cleans dataset: ensures valid target and sets appropriate data types."""
    clean_df = df.copy()

    # Target must be a positive number
    clean_df[target_col] = pd.to_numeric(clean_df[target_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[target_col])
    clean_df = clean_df[clean_df[target_col] > 0]

    # Cast numeric columns
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Cast categorical columns as clean strings
    for col in categorical_cols:
        clean_df[col] = clean_df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    return clean_df.reset_index(drop=True)

# =========================================================
# 3. MODEL TRAINING: RANDOM FOREST PIPELINE
# =========================================================
def train_random_forest(df, target_col, numeric_cols, categorical_cols):
    """
    Trains a Random Forest Regressor using a clean ColumnTransformer pipeline:
    - Numeric: Imputed with median
    - Categorical: Imputed with mode + One-Hot Encoded
    """
    features = numeric_cols + categorical_cols
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipelines
    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Random Forest Regressor (100 decision trees)
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", rf)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    # Feature Importance
    rf_model = model.named_steps["regressor"]
    feature_names = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if categorical_cols:
        ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(ohe.get_feature_names_out(categorical_cols).tolist())

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance (%)": (rf_model.feature_importances_ * 100).round(2)
    }).sort_values(by="Importance (%)", ascending=False).reset_index(drop=True)

    eval_df = pd.DataFrame({
        "Actual (min)": y_test.to_numpy(),
        "Predicted (min)": predictions.round(1),
        "Error (min)": np.abs(y_test.to_numpy() - predictions).round(1)
    })

    return model, {"MAE": mae, "RMSE": rmse, "R²": r2}, eval_df, importance_df, len(X_train), len(X_test)

# =========================================================
# 4. SIDEBAR & DATA INGESTION
# =========================================================
st.markdown('<div class="hero"><h1>🚚 DeliveryIQ</h1><p>Food Delivery Duration Prediction · Random Forest Regression · AI Applications (Section 413 B)</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("DeliveryIQ")
    page = st.radio("Navigation", ["Overview", "Predict delivery", "Dataset & EDA", "Model evaluation"])
    st.divider()

    st.subheader("Data Source")
    uploaded = st.file_uploader("Upload CSV File", type=["csv"], help="Upload any delivery dataset. Columns are detected dynamically.")

    # Default file path
    csv_path = Path("data/Food_Delivery_Times.csv")
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        data_source = f"Uploaded: {uploaded.name}"
    elif csv_path.exists():
        raw_df = pd.read_csv(csv_path)
        data_source = f"Dataset: {csv_path.name}"
    else:
        st.error("No dataset found. Please upload a CSV.")
        st.stop()

    # Dynamic Column Detection
    auto_target, auto_ids, auto_nums, auto_cats = detect_columns(raw_df)

    with st.expander("⚙️ Column Settings", expanded=False):
        all_cols = raw_df.columns.tolist()
        target_col = st.selectbox("Target Column", options=all_cols, index=all_cols.index(auto_target) if auto_target in all_cols else len(all_cols)-1)
        remaining = [c for c in all_cols if c != target_col and c not in auto_ids]
        numeric_cols = st.multiselect("Numeric Features", options=remaining, default=[c for c in auto_nums if c in remaining])
        categorical_cols = [c for c in remaining if c not in numeric_cols]

    # Clean and validate dataset
    data = clean_dataset(raw_df, target_col, numeric_cols, categorical_cols)
    if len(data) < 10:
        st.error("Dataset has fewer than 10 valid records.")
        st.stop()

    st.divider()
    st.caption("PROJECT TEAM")
    st.write("**Lavish** (26BCS11757)")
    st.write("**Naseeb** (26BCS11720)")
    st.write("**Sangam Saini** (26BCS11705)")
    st.write("**Aryan Kumar** (26BCS11744)")

# Train the Random Forest Model
model, metrics, eval_df, importance_df, n_train, n_test = train_random_forest(data, target_col, numeric_cols, categorical_cols)
st.caption(f"📁 **{data_source}** · **{len(data):,}** records · Target: `{target_col}` · Algorithm: **Random Forest (100 Trees)**")

# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if page == "Overview":
    st.subheader("Project Overview")
    st.write(
        "DeliveryIQ is a machine learning solution designed to predict food delivery durations. "
        "It uses an ensemble **Random Forest Regressor** to model complex non-linear relationships between route distance, "
        "traffic intensity, weather conditions, order preparation time, vehicle type, and courier experience."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(data):,}")
    c2.metric(f"Average {target_col}", f"{data[target_col].mean():.1f} min")
    c3.metric("Test MAE", f"{metrics['MAE']:.2f} min", help="Mean Absolute Error on holdout test set.")
    c4.metric("Test R² Score", f"{metrics['R²']:.3f}", help="Variance explained by Random Forest.")

    st.divider()
    st.subheader("Detected Data Schema")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.write("**🎯 Target Variable:**")
        st.markdown(f"<span class='tag'><b>{target_col}</b></span>", unsafe_allow_html=True)
        if auto_ids:
            st.write("**🆔 Excluded ID Columns:**")
            st.markdown(" ".join([f"<span class='tag'>{c}</span>" for c in auto_ids]), unsafe_allow_html=True)
    with s2:
        st.write(f"**🔢 Numeric Features ({len(numeric_cols)}):**")
        st.markdown(" ".join([f"<span class='tag'>{c}</span>" for c in numeric_cols]) if numeric_cols else "None", unsafe_allow_html=True)
    with s3:
        st.write(f"**🏷️ Categorical Features ({len(categorical_cols)}):**")
        st.markdown(" ".join([f"<span class='tag'>{c}</span>" for c in categorical_cols]) if categorical_cols else "None", unsafe_allow_html=True)

    st.divider()
    st.subheader("Why Random Forest?")
    st.markdown("""
    - **Non-Linear Relationships:** Delivery times rarely follow a straight line (e.g. bad weather and heavy traffic interact to multiply delays).
    - **Ensemble Averaging:** Combines 100 decision trees to minimize overfitting and reduce prediction variance.
    - **Feature Importance:** Quantifies exactly how much each feature influences the estimated delivery duration.
    """)

# =========================================================
# PAGE 2: PREDICT DELIVERY
# =========================================================
elif page == "Predict delivery":
    st.subheader("Live Delivery Time Prediction")
    st.write("Enter the delivery conditions below to generate an instant duration estimate using the trained Random Forest.")

    with st.form("prediction_form"):
        inputs = {}
        all_features = numeric_cols + categorical_cols
        mid = (len(all_features) + 1) // 2
        col_left, col_right = st.columns(2)

        for i, feat in enumerate(all_features):
            container = col_left if i < mid else col_right
            with container:
                label = feat.replace("_", " ").title()
                if feat in numeric_cols:
                    s = data[feat].dropna()
                    min_val, max_val, med_val = float(s.min()), float(s.max()), float(s.median())
                    is_int = (s % 1 == 0).all()
                    if is_int:
                        inputs[feat] = st.slider(label, int(min_val), int(max_val), int(round(med_val)), step=1)
                    else:
                        inputs[feat] = st.slider(label, round(min_val, 1), round(max_val, 1), round(med_val, 1), step=0.1)
                else:
                    options = sorted(data[feat].dropna().astype(str).unique().tolist())
                    mode_val = data[feat].dropna().astype(str).mode().iloc[0] if options else ""
                    default_idx = options.index(mode_val) if mode_val in options else 0
                    inputs[feat] = st.selectbox(label, options=options, index=default_idx)

        submit = st.form_submit_button("Predict Delivery Time", type="primary", use_container_width=True)

    if submit:
        input_row = pd.DataFrame([inputs])
        predicted_time = float(model.predict(input_row)[0])
        lo_bound = max(1, round(predicted_time - metrics["MAE"]))
        hi_bound = round(predicted_time + metrics["MAE"])

        res1, res2 = st.columns(2)
        res1.metric("Predicted Delivery Time", f"{predicted_time:.0f} minutes")
        res2.metric("Planning Interval (± MAE)", f"{lo_bound} – {hi_bound} min")
        st.success(f"Estimated arrival: **{predicted_time:.0f} minutes** under these conditions.")

# =========================================================
# PAGE 3: DATASET & EDA
# =========================================================
elif page == "Dataset & EDA":
    st.subheader("Dataset Preview")
    st.dataframe(data.head(50), use_container_width=True)
    st.download_button("📥 Download Active CSV", data=data.to_csv(index=False).encode("utf-8"), file_name="delivery_data.csv", mime="text/csv")

    st.divider()
    st.subheader("Descriptive Statistics")
    stat_cols = numeric_cols + [target_col]
    if stat_cols:
        st.dataframe(data[stat_cols].describe().round(2), use_container_width=True)

    st.divider()
    st.subheader("Exploratory Visualizations")
    v1, v2 = st.columns(2)
    with v1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data[target_col], kde=True, ax=ax, color="#6366f1")
        ax.set_title(f"Distribution of {target_col.replace('_', ' ').title()}")
        ax.set_xlabel(f"{target_col.replace('_', ' ').title()} (minutes)")
        st.pyplot(fig)
        plt.close(fig)

    with v2:
        if categorical_cols:
            cat_choice = st.selectbox("Compare Delivery Time by:", options=categorical_cols)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=data, x=cat_choice, y=target_col, ax=ax, palette="mako")
            ax.set_title(f"Delivery Time across {cat_choice.replace('_', ' ').title()}")
            plt.xticks(rotation=20)
            st.pyplot(fig)
            plt.close(fig)

    # Correlation Matrix
    if len(stat_cols) > 1:
        st.divider()
        st.subheader("Numeric Correlation Matrix")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(data[stat_cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

# =========================================================
# PAGE 4: MODEL EVALUATION
# =========================================================
elif page == "Model evaluation":
    st.subheader("Random Forest Model Evaluation")
    st.write(f"Training Samples: **{n_train:,}** · Test Samples: **{n_test:,}** · Split: **80/20**")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.2f} min", help="Mean Absolute Error (Average deviation).")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f} min", help="Root Mean Squared Error.")
    m3.metric("R² Score", f"{metrics['R²']:.3f}", help="Coefficient of determination.")

    st.divider()
    st.subheader("Feature Importance Breakdown")
    st.write("Random Forest calculates the contribution percentage of each feature toward the prediction:")
    st.dataframe(importance_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(importance_df) * 0.35)))
    sorted_imp = importance_df.sort_values(by="Importance (%)", ascending=True)
    ax.barh(sorted_imp["Feature"], sorted_imp["Importance (%)"], color="#6366f1")
    ax.set_xlabel("Importance (%)")
    ax.set_title("Random Forest: Feature Importance (%)")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Actual vs. Predicted Delivery Time")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=eval_df, x="Actual (min)", y="Predicted (min)", ax=ax, color="#6366f1", alpha=0.7)
    lims = [min(eval_df["Actual (min)"].min(), eval_df["Predicted (min)"].min()), max(eval_df["Actual (min)"].max(), eval_df["Predicted (min)"].max())]
    ax.plot(lims, lims, linestyle="--", color="#f59e0b", label="Perfect Prediction Line")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Holdout Predictions Table")
    st.dataframe(eval_df.head(50), use_container_width=True, hide_index=True)
    st.download_button("📥 Download Holdout Predictions", data=eval_df.to_csv(index=False).encode("utf-8"), file_name="rf_predictions.csv", mime="text/csv")
