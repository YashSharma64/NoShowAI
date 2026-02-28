import os
from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, r2_score

# Allow large risk tables to be styled (default 262144 cells).
pd.set_option("styler.render.max_elements", 2_000_000)


@dataclass
class ModelBundle:
    model: object
    scaler: Optional[object]


APP_TITLE = "Clinical Appointment No-Show Prediction Dashboard"
APP_SUBTITLE = (
    "Predicts appointment attendance risk and highlights high-risk patients for proactive intervention."
)

DEFAULT_MODEL_PATH = os.path.join("model", "best_model.pkl")
DEFAULT_SCALER_PATH = os.path.join("model", "scaler.pkl")


def _inject_css() -> None:
    st.markdown(
        """
<style>
/* Subtle, minimal healthcare dashboard styling */
:root {
  --ns-border: rgba(100, 80, 60, 0.25);
  --ns-bg-soft: rgba(14, 118, 200, 0.08);
  --ns-green: rgba(34, 197, 94, 0.18);
  --ns-red: rgba(239, 68, 68, 0.18);
  --ns-text:#DAA520;
  --ns-text-muted: #5a4a38;
}

/* Canvas + layout */
.stApp {
  background: #f7eddc;
}

.stMainBlockContainer, .block-container {
  padding-top: 3.2rem;
  padding-bottom: 3rem;
  max-width: 1120px;
}

.ns-shell {
  background: #fbf3e3;
  border-radius: 18px;
  border: 1px solid rgba(148, 114, 80, 0.30);
  padding: 26px 32px 30px 32px;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
}

.ns-shell-inner {
  background: #ffffff;
  border-radius: 16px;
  border: 1px solid rgba(0, 0, 0, 0.06);
  padding: 20px 20px 24px 20px;
}

/* Typography + chrome cleanup */
html, body, [class*="css"] {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

/* Make all text inside the app dark and readable */
.stApp p, .stApp span, .stApp div, .stApp label,
.stApp .stMarkdown, .stApp .stText {
  color: var(--ns-text);
}

header[data-testid="stHeader"] {
  background: transparent;
}

/* Hide only the action items in toolbar to keep sidebar toggle visible */
[data-testid="stToolbar"] [data-testid="stActionMenu"] {
  display: none !important;
}
[data-testid="stToolbar"] button[kind="secondary"] {
  display: none !important; /* Hide the Deploy button */
}

footer {
  visibility: hidden;
  height: 0;
}

section[data-testid="stSidebar"] {
  background: #eddcbf;
  border-right: 1px solid rgba(148, 114, 80, 0.25);
}

/* Sidebar — scope text colours to actual text nodes only */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stMarkdown {
  color: #2c2018 !important;
}

/* Radio button labels */
section[data-testid="stSidebar"] .stRadio label p {
  color: #3d2e1e !important;
  font-weight: 500;
}

/* Selected radio item highlight */
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
  color: #2c2018 !important;
}

/* Sidebar collapse / toggle button — keep it visible */
button[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"] {
  background: #d9c4a0 !important;
  border-radius: 8px !important;
}
button[data-testid="collapsedControl"] svg,
button[data-testid="baseButton-headerNoPadding"] svg {
  fill: #2c2018 !important;
  stroke: #2c2018 !important;
}

/* Sidebar close arrow */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
  background: rgba(148, 114, 80, 0.15) !important;
  border-radius: 8px;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg {
  fill: #2c2018 !important;
  stroke: #2c2018 !important;
}

/* Divider line in sidebar */
section[data-testid="stSidebar"] hr {
  border-color: rgba(100, 80, 60, 0.25) !important;
}

button[kind="primary"] {
  border-radius: 999px !important;
}

.ns-card {
  border: 1px solid rgba(100, 80, 60, 0.2);
  border-radius: 14px;
  padding: 14px 14px;
  background: white;
  color: var(--ns-text);
  transition: box-shadow 180ms ease, transform 180ms ease;
}
.ns-card:hover {
  box-shadow: 0 10px 28px rgba(0,0,0,0.08);
  transform: translateY(-1px);
}

.ns-kpi {
  border: 1px solid rgba(100, 80, 60, 0.22);
  border-radius: 14px;
  padding: 12px 14px;
  background: linear-gradient(160deg, #e8f4fd, #ffffff);
  color: var(--ns-text);
}

.ns-kpi b {
  color: #1e3a5f;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.ns-helper {
  color: var(--ns-text-muted);
  font-size: 0.92rem;
}

.ns-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(100, 80, 60, 0.30);
  background: rgba(255,255,255,0.6);
  color: var(--ns-text-muted);
  font-size: 0.85rem;
}

/* ── File uploader ──────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: #ffffff !important;
  border: 2px dashed rgba(100, 80, 60, 0.35) !important;
  border-radius: 14px !important;
  padding: 12px 16px !important;
}
[data-testid="stFileUploader"] * {
  color: var(--ns-text) !important;
}
[data-testid="stFileUploader"] small {
  color: var(--ns-text-muted) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  color: var(--ns-text) !important;
}

/* ── Alert / info / success / warning boxes ─────────────────── */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  border: 1px solid rgba(100, 80, 60, 0.18) !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
  color: #1e293b !important;
}

/* ── Dataframe / table ──────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(100, 80, 60, 0.15);
}

/* ── Subheader text ─────────────────────────────────────────── */
.stApp h2, .stApp h3 {
  color: #2c2018 !important;
}

/* ── Expander ───────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: #ffffff;
  border: 1px solid rgba(100, 80, 60, 0.18);
  border-radius: 12px;
}
[data-testid="stExpander"] * {
  color: var(--ns-text);
}

/* ── Input fields ───────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: #fffaf5 !important;
  border: 1px solid rgba(100, 80, 60, 0.28) !important;
  border-radius: 8px !important;
  color: var(--ns-text) !important;
}

/* ── Slider ─────────────────────────────────────────────────── */
[data-testid="stSlider"] * {
  color: var(--ns-text) !important;
}

/* ── Spinner ────────────────────────────────────────────────── */
[data-testid="stSpinner"] * {
  color: var(--ns-text) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    # Top row: menu icon + title stack, right-aligned pill
    c_icon, c_title, c_pill = st.columns([0.08, 0.72, 0.20], vertical_alignment="center")
    with c_icon:
        st.markdown(
            "<div style='font-size:22px;font-weight:500;letter-spacing:2px;'>≡</div>",
            unsafe_allow_html=True,
        )
    with c_title:
        st.markdown(
            "<div style='font-size:32px;font-weight:650;letter-spacing:0.12em;"
            "text-transform:uppercase;margin-bottom:6px;color:#4b3b2a;'>NoShow.AI</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:16px;font-weight:550;color:rgba(55,65,81,0.96);'>{APP_TITLE}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:13px;color:rgba(75,85,99,0.9);margin-top:4px;'>{APP_SUBTITLE}</div>",
            unsafe_allow_html=True,
        )
    with c_pill:
        st.markdown(
            "<div style='display:flex;justify-content:flex-end;'>"
            "<div class='ns-pill'>Clinical Operations • Live</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def _read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def _safe_load_bundle(
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
) -> Tuple[Optional[ModelBundle], Optional[str]]:
    if not os.path.exists(model_path):
        return None, f"Model not found at `{model_path}`. Place your trained `.pkl` there (e.g. from the project root: `model/best_model.pkl`). Using demo predictions until then."

    try:
        model = joblib.load(model_path)
    except Exception as e:
        return None, f"Failed to load model from `{model_path}` ({type(e).__name__}). Using dummy predictions."

    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None

    return ModelBundle(model=model, scaler=scaler), None


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Common columns seen in training code
    x = x.drop(columns=["NoShow", "ScheduledDay", "AppointmentDay"], errors="ignore")

    # Basic cleanup: convert booleans to ints, keep numeric when possible
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(int)

    # One-hot encode categoricals in a safe default manner
    cat_cols = [c for c in x.columns if x[c].dtype == object]
    if cat_cols:
        x = pd.get_dummies(x, columns=cat_cols, drop_first=False)

    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    return x


def _predict(df: pd.DataFrame, bundle: Optional[ModelBundle]) -> Tuple[pd.Series, pd.Series]:
    x = _prepare_features(df)

    if bundle is None:
        # Dummy logic: deterministic pseudo-risk based on row index.
        probs = (pd.Series(range(len(df))) % 100) / 100
        labels = (probs >= 0.5).map({True: "No-Show", False: "Show"})
        return labels, probs

    model = bundle.model
    scaler = bundle.scaler

    # Align columns to what the model saw during training when available.
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        x = x.reindex(columns=list(feature_names), fill_value=0)

    x_input = x
    if scaler is not None:
        try:
            x_input = scaler.transform(x)
        except Exception:
            x_input = x

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_input)
        # Convention: class 1 is No-Show in training code.
        if proba.shape[1] >= 2:
            probs = pd.Series(proba[:, 1], index=df.index)
        else:
            probs = pd.Series(proba[:, 0], index=df.index)
    else:
        preds = model.predict(x_input)
        probs = pd.Series(preds, index=df.index).astype(float).clip(0, 1)

    preds = (probs >= 0.5).astype(int)
    labels = pd.Series(preds, index=df.index).map({1: "No-Show", 0: "Show"})
    return labels, probs


def _style_risk_table(high_risk_threshold: float):
    def _row_style(row: pd.Series):
        prob = float(row.get("probability", 0.0))
        if prob >= high_risk_threshold:
            return ["background-color: rgba(239, 68, 68, 0.14)"] * len(row)
        return ["background-color: rgba(34, 197, 94, 0.10)"] * len(row)

    return _row_style


def page_upload() -> None:
    st.subheader("Upload Dataset")
    st.markdown(
        "<div class='ns-helper'><b>Step 1 of 3</b> — Upload the hospital’s <i>preprocessed</i> appointment file to begin the no-show risk workflow.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ns-helper' style='margin-top:4px;'>This dataset should already include cleaned dates, lead-time, encoded demographics, and reminder information from your preprocessing pipeline.</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

    if uploaded is None:
        st.info("Upload a CSV to begin. If you don't have one handy, you can use your processed dataset file.")
        return

    with st.spinner("Loading dataset..."):
        df = _read_csv(uploaded)

    st.session_state["df"] = df
    st.success("Dataset uploaded successfully.")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("<div class='ns-kpi'><b>Rows</b><br/>{}</div>".format(df.shape[0]), unsafe_allow_html=True)
    with k2:
        st.markdown("<div class='ns-kpi'><b>Columns</b><br/>{}</div>".format(df.shape[1]), unsafe_allow_html=True)
    with k3:
        st.markdown("<div class='ns-kpi'><b>Status</b><br/>Ready</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='ns-card'><b>Preview</b></div>", unsafe_allow_html=True)
    st.dataframe(df.head(20), width="stretch")


def page_risk_dashboard() -> None:
    st.subheader("Risk Dashboard")
    st.markdown(
        "<div class='ns-helper'><b>Step 2 of 3</b> — Run the prediction engine to score each upcoming appointment for no-show risk.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ns-helper' style='margin-top:4px;'>High‑risk rows surface patients who may benefit from reminders, rescheduling, or social‑work follow‑up.</div>",
        unsafe_allow_html=True,
    )

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first from the sidebar.")
        return

    df = st.session_state["df"].copy()

    with st.expander("Model Settings", expanded=False):
        model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
        scaler_path = st.text_input("Scaler path (optional)", value=DEFAULT_SCALER_PATH)
        high_risk_threshold = st.slider("High-risk threshold", 0.50, 0.95, 0.70, 0.05)

    bundle, warn = _safe_load_bundle(model_path=model_path, scaler_path=scaler_path)
    if warn is not None:
        st.info(warn)

    c1, c2 = st.columns([0.35, 0.65], vertical_alignment="center")
    with c1:
        run = st.button("Predict No-Show Risk", type="primary", use_container_width=True)
    with c2:
        st.markdown(
            "<div class='ns-helper'>Highlighted rows indicate appointments at elevated risk of no‑show and may require proactive outreach.</div>",
            unsafe_allow_html=True,
        )

    if not run and "predictions" not in st.session_state:
        st.markdown("<div class='ns-card'>No predictions yet. Click <b>Predict No-Show Risk</b>.</div>", unsafe_allow_html=True)
        return

    if run:
        with st.spinner("Scoring appointments..."):
            labels, probs = _predict(df, bundle)

        results = df.copy()
        results["prediction"] = labels.values
        results["probability"] = probs.values

        st.session_state["predictions"] = results
        st.toast("Risk scoring complete", icon=None)

    results = st.session_state["predictions"].copy()

    high_risk = (results["probability"] >= high_risk_threshold).sum() if "probability" in results else 0
    total = len(results)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            "<div class='ns-kpi'><b>Total Appointments</b><br/>{}</div>".format(total),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            "<div class='ns-kpi'><b>High Risk</b><br/>{}</div>".format(int(high_risk)),
            unsafe_allow_html=True,
        )
    with k3:
        pct = (high_risk / total * 100.0) if total else 0.0
        st.markdown(
            "<div class='ns-kpi'><b>High-Risk Rate</b><br/>{:.1f}%</div>".format(pct),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("<div class='ns-card'><b>Predicted Risk Table</b></div>", unsafe_allow_html=True)

    n_cells = results.shape[0] * results.shape[1]
    # For very large tables, avoid Pandas Styler entirely (it has hard limits and can be slow).
    max_cells_styled = 200_000
    use_styler = "probability" in results.columns and n_cells <= max_cells_styled

    if use_styler:
        styled = results.style.apply(_style_risk_table(high_risk_threshold), axis=1)
        st.dataframe(styled, width="stretch", height=520)
    else:
        if "probability" in results.columns and n_cells > max_cells_styled:
            st.caption(
                "Table is large; showing without row highlighting. "
                "Sort or filter by the **probability** column to focus on high‑risk appointments."
            )
        st.dataframe(results, width="stretch", height=520)


def _compute_model_metrics(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute Accuracy, F1 (No-Show), R² for each available model.

    Strategy (in priority order):
    1. Load individual model pkl files and score against ground truth.
    2. Fall back to best_model.pkl if individual pkls are missing.
    3. Use session-state predictions when no pkl files exist at all.
    """
    if "NoShow" not in df.columns:
        return None

    y_true = df["NoShow"].astype(int).values
    X = _prepare_features(df)

    scaler = None
    if os.path.exists("model/scaler.pkl"):
        try:
            scaler = joblib.load("model/scaler.pkl")
        except Exception:
            pass

    def _score_model(mdl, name: str, use_scaler: bool = False):
        feat = getattr(mdl, "feature_names_in_", None)
        X_use = X.reindex(columns=list(feat), fill_value=0) if feat is not None else X
        if use_scaler and scaler is not None:
            try:
                X_use = scaler.transform(X_use)
            except Exception:
                pass
        y_pred = mdl.predict(X_use)
        return {
            "Accuracy":     round(accuracy_score(y_true, y_pred), 4),
            "F1 (No-Show)": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
            "R² Score":     round(r2_score(y_true, y_pred), 4),
            "_missing":     False,
        }

    model_files = {
        "Logistic Regression": ("model/logistic_regression.pkl", True),
        "Decision Tree":       ("model/decision_tree.pkl",        False),
        "Random Forest":       ("model/random_forest.pkl",        False),
    }

    rows = []
    any_individual_found = False

    for name, (path, use_scaler) in model_files.items():
        if os.path.exists(path):
            any_individual_found = True
            try:
                mdl = joblib.load(path)
                row = _score_model(mdl, name, use_scaler)
                row["Model"] = name
                rows.append(row)
            except Exception as e:
                rows.append({"Model": name, "Accuracy": "—", "F1 (No-Show)": "—",
                             "R² Score": str(e)[:40], "_missing": True})
        else:
            rows.append({"Model": name, "Accuracy": "—", "F1 (No-Show)": "—",
                         "R² Score": "—", "_missing": True})

    # ── Fallback 1: best_model.pkl ─────────────────────────────────────────
    if not any_individual_found and os.path.exists("model/best_model.pkl"):
        try:
            mdl = joblib.load("model/best_model.pkl")
            row = _score_model(mdl, "Best Model", use_scaler=False)
            row["Model"] = "Best Model (saved)"
            return pd.DataFrame([row])
        except Exception:
            pass

    # ── Fallback 2: session-state predictions ─────────────────────────────
    if not any_individual_found and "predictions" in st.session_state:
        preds = st.session_state["predictions"]
        if "prediction" in preds.columns and "probability" in preds.columns:
            # Map string labels back to int
            y_pred_str = preds["prediction"].reindex(df.index)
            y_pred = y_pred_str.map({"No-Show": 1, "Show": 0}).fillna(0).astype(int).values
            # Align lengths
            min_len = min(len(y_true), len(y_pred))
            y_t, y_p = y_true[:min_len], y_pred[:min_len]
            return pd.DataFrame([{
                "Model":        "Active Model (session)",
                "Accuracy":     round(accuracy_score(y_t, y_p), 4),
                "F1 (No-Show)": round(f1_score(y_t, y_p, pos_label=1, zero_division=0), 4),
                "R² Score":     round(r2_score(y_t, y_p), 4),
                "_missing":     False,
            }])

    return pd.DataFrame(rows)


def page_analytics() -> None:
    st.subheader("Analytics")
    st.markdown(
        "<div class='ns-helper'><b>Step 3 of 3</b> — Review system‑level behaviour and model accuracy metrics.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ns-helper' style='margin-top:4px;'>Explore prediction distributions, risk calibration, and per-model R² / F1 scores below.</div>",
        unsafe_allow_html=True,
    )

    if "predictions" not in st.session_state:
        st.info("Run predictions first to unlock analytics.")
        return

    results = st.session_state["predictions"].copy()

    # ── Model Performance ────────────────────────────────────────────────────
    st.markdown("### 📊 Model Performance")
    df_source = st.session_state.get("df")
    if df_source is not None and "NoShow" in df_source.columns:
        with st.spinner("Computing model metrics…"):
            metrics_df = _compute_model_metrics(df_source)

        if metrics_df is not None:
            display_df = metrics_df.drop(columns=["_missing"], errors="ignore")

            # Highlight best R² row
            def _highlight_best(row):
                try:
                    r2_val = float(row["R² Score"])
                    best_r2 = metrics_df[metrics_df["_missing"] == False]["R² Score"].astype(float).max()
                    if r2_val == best_r2:
                        return ["background-color: rgba(59,130,246,0.12)"] * len(row)
                except Exception:
                    pass
                return [""] * len(row)

            styled_metrics = display_df.style.apply(_highlight_best, axis=1).format(
                {"Accuracy": lambda v: f"{v:.4f}" if isinstance(v, float) else v,
                 "F1 (No-Show)": lambda v: f"{v:.4f}" if isinstance(v, float) else v,
                 "R² Score": lambda v: f"{v:.4f}" if isinstance(v, float) else v}
            )
            st.dataframe(styled_metrics, width="stretch", hide_index=True)

            # KPI strip for best model's R²
            valid = metrics_df[metrics_df["_missing"] == False]
            if not valid.empty:
                best_row = valid.loc[valid["R² Score"].astype(float).idxmax()]
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.markdown(
                        f"<div class='ns-kpi'><b>Best Model</b><br/>{best_row['Model']}</div>",
                        unsafe_allow_html=True,
                    )
                with k2:
                    st.markdown(
                        f"<div class='ns-kpi'><b>R² Score</b><br/>{float(best_row['R² Score']):.4f}</div>",
                        unsafe_allow_html=True,
                    )
                with k3:
                    st.markdown(
                        f"<div class='ns-kpi'><b>Accuracy</b><br/>{float(best_row['Accuracy']):.4f}</div>",
                        unsafe_allow_html=True,
                    )
                with k4:
                    st.markdown(
                        f"<div class='ns-kpi'><b>F1 (No-Show)</b><br/>{float(best_row['F1 (No-Show)']):.4f}</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Upload a dataset with a `NoShow` column on the Upload page to see per-model R² scores.")

    st.markdown("---")

    # ── Charts ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='ns-card'><b>No-Show Distribution</b></div>", unsafe_allow_html=True)
        if "prediction" in results.columns:
            counts = results["prediction"].value_counts().reindex(["Show", "No-Show"]).fillna(0)
            st.bar_chart(counts)
        else:
            st.info("Prediction column not found.")

    with col2:
        st.markdown("<div class='ns-card'><b>Risk Probability Histogram</b></div>", unsafe_allow_html=True)
        if "probability" in results.columns:
            hist = np.histogram(results["probability"].astype(float).clip(0, 1), bins=12, range=(0, 1))[0]
            st.bar_chart(pd.Series(hist, index=[f"{i/12:.2f}" for i in range(12)]))
        else:
            st.info("Probability column not found.")

    st.markdown("---")
    st.markdown("<div class='ns-card'><b>🔍 Feature Importance</b></div>", unsafe_allow_html=True)

    # 1. Try saved tree model pkl files first
    _fi_shown = False
    for _mdl_path in ["model/random_forest.pkl", "model/decision_tree.pkl", "model/best_model.pkl"]:
        if os.path.exists(_mdl_path):
            try:
                _mdl = joblib.load(_mdl_path)
                if hasattr(_mdl, "feature_importances_") and hasattr(_mdl, "feature_names_in_"):
                    fi = pd.Series(
                        _mdl.feature_importances_,
                        index=_mdl.feature_names_in_,
                    ).sort_values(ascending=False).head(15)
                    st.bar_chart(fi)
                    st.caption(f"Top-15 feature importances from `{_mdl_path}`.")
                    _fi_shown = True
                    break
            except Exception:
                pass

    # 2. Fall back: train a tiny RF on-the-fly from the uploaded data
    if not _fi_shown:
        df_src = st.session_state.get("df")
        if df_src is not None and "NoShow" in df_src.columns:
            try:
                from sklearn.ensemble import RandomForestClassifier as _RFC
                _X = _prepare_features(df_src)
                _y = df_src["NoShow"].astype(int)
                # Keep only numeric, drop constant columns
                _X = _X.select_dtypes(include=[np.number]).loc[:, _X.nunique() > 1]
                # Use a small fast forest (no need to save it)
                _quick_rf = _RFC(n_estimators=50, max_depth=8, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
                _quick_rf.fit(_X, _y)
                fi = pd.Series(
                    _quick_rf.feature_importances_,
                    index=_X.columns,
                ).sort_values(ascending=False).head(15)
                st.bar_chart(fi)
                st.caption("Top-15 feature importances — computed on-the-fly from your uploaded data.")
                _fi_shown = True
            except Exception as e:
                st.caption(f"Could not compute feature importance: {e}")

    if not _fi_shown:
        st.info("Upload a labelled dataset (with `NoShow` column) to see feature importances.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _inject_css()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Upload Dataset", "Risk Dashboard", "Analytics"],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Model: `model/best_model.pkl` (default)")
    st.sidebar.markdown(
        "<div class='ns-helper' style='margin-top:10px; font-size:0.8rem;'>"
        "<b>Flow</b><br/>Raw data → Preprocessing → Model training → "
        "<b>Upload</b> → <b>Risk dashboard</b> → <b>Analytics</b>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Header outside any shell wrapper to avoid empty-box glitch
    _header()
    st.markdown(
        "<hr style='margin:14px 0 18px 0;border:none;border-top:1px solid rgba(100,80,60,0.2);'/>",
        unsafe_allow_html=True,
    )

    if page == "Upload Dataset":
        page_upload()
    elif page == "Risk Dashboard":
        page_risk_dashboard()
    else:
        page_analytics()


if __name__ == "__main__":
    main()
