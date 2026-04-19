from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


DROP_MODEL_COLUMNS = ["NoShow", "ScheduledDay", "AppointmentDay"]


def clean_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully empty rows, then fill missing values safely by dtype."""
    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    # Drop unusable rows that are entirely empty.
    cleaned = cleaned.dropna(how="all")

    for col in cleaned.columns:
        series = cleaned[col]

        if pd.api.types.is_bool_dtype(series):
            cleaned[col] = series.astype(int).fillna(0)
            continue

        if pd.api.types.is_numeric_dtype(series):
            median_val = series.median()
            fill_val = 0 if pd.isna(median_val) else median_val
            cleaned[col] = series.fillna(fill_val)
            continue

        mode_vals = series.mode(dropna=True)
        fill_val = "Unknown" if mode_vals.empty else mode_vals.iloc[0]
        cleaned[col] = series.fillna(fill_val)

    return cleaned


def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready features from cleaned input data."""
    features = clean_input_data(df)
    features = features.drop(columns=DROP_MODEL_COLUMNS, errors="ignore")

    for col in features.columns:
        if features[col].dtype == bool:
            features[col] = features[col].astype(int)

    cat_cols = [c for c in features.columns if features[c].dtype == object]
    if cat_cols:
        features = pd.get_dummies(features, columns=cat_cols, drop_first=False)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


def _humanize_factor(feature_name: str) -> str:
    name = feature_name.replace("_", " ").strip()
    lower = name.lower()

    if "waiting" in lower or "lead" in lower:
        return "Long lead time"
    if "noshow" in lower or "no show" in lower or "previous" in lower:
        return "Past no-shows"
    if "sms" in lower or "reminder" in lower:
        return "Reminder history"
    if "age" in lower:
        return "Age-related risk"
    if "distance" in lower or "neighbour" in lower:
        return "Access or travel constraints"
    if "diab" in lower or "hipert" in lower or "hypertension" in lower:
        return "Chronic condition burden"

    return name.title()


def extract_top_factors(
    risk: float,
    row_features: pd.Series,
    model: object,
    top_n: int = 2,
) -> List[str]:
    """Extract top contributing factors for one prediction row."""
    feature_names = list(row_features.index)
    values = row_features.astype(float).values
    contributions = np.abs(values)

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2 and coef.shape[0] >= 1:
            coef = coef[0]
        if coef.shape[0] == len(values):
            contributions = np.abs(coef * values)
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        if importances.shape[0] == len(values):
            contributions = np.abs(importances * values)

    top_idx = np.argsort(contributions)[::-1][:top_n]

    factors: List[str] = []
    for idx in top_idx:
        if contributions[idx] <= 0:
            continue
        factors.append(_humanize_factor(feature_names[idx]))

    if not factors:
        if risk >= 0.75:
            return ["Long lead time", "Past no-shows"]
        if risk >= 0.5:
            return ["Moderate attendance risk"]
        return ["Low attendance risk"]

    # Keep ordering while removing duplicates from humanized names.
    deduped = []
    seen = set()
    for item in factors:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def build_structured_input(
    risk: float,
    factors: List[str],
    guidelines: List[str] | None = None,
) -> dict:
    payload = {
        "risk": round(float(risk), 4),
        "factors": factors,
    }
    if guidelines is not None:
        payload["guidelines"] = guidelines
    return payload