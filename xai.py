"""
Explainable AI (SHAP) for diabetes prediction.

Supports:
- Flask/app_xai.py: explain_prediction(input_data, model=..., preprocessing_params=..., feature_names=...)
- Pipelines: explain_sample_shap(model, X_df, feature_names)

SHAP is optional at import time; if missing, explanations degrade gracefully.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore

MODEL_FILENAME = "diabetes-prediction-xai-model.pkl"
PREPROCESSING_FILENAME = "preprocessing_params.pkl"
FEATURE_NAMES_FILENAME = "feature_names.pkl"
VISUALIZATIONS_DIR = "xai_visualizations"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


def load_model_and_data() -> Tuple[Any, Dict[str, Any], List[str]]:
    """Load trained model, preprocessing params, and feature names from disk."""
    try:
        model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model file '{MODEL_FILENAME}' not found. Train with: python model.py"
        ) from exc

    try:
        preprocessing_params = joblib.load(PREPROCESSING_FILENAME)
    except FileNotFoundError:
        preprocessing_params = {
            "glucose_mean": 121.0,
            "bloodpressure_mean": 72.0,
            "skinthickness_median": 29.0,
            "insulin_median": 102.5,
            "bmi_median": 32.0,
        }

    try:
        feature_names = joblib.load(FEATURE_NAMES_FILENAME)
    except FileNotFoundError:
        feature_names = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DPF",
            "Age",
        ]

    return model, preprocessing_params, list(feature_names)


def preprocess_input(
    input_data: Any,
    preprocessing_params: Dict[str, Any],
    feature_names: Sequence[str],
) -> np.ndarray:
    """
    Apply the same zero-imputation rules as training (dict or array → 2D array).
    """
    if isinstance(input_data, dict):
        processed = [input_data.get(f, 0) for f in feature_names]
        input_array = np.array([processed], dtype=float)
    else:
        input_array = np.asarray(input_data, dtype=float)
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)

    out = input_array.copy()
    for i, feature in enumerate(feature_names):
        if i >= out.shape[1]:
            break
        if float(out[0, i]) == 0.0:
            if feature == "Glucose" and "glucose_mean" in preprocessing_params:
                out[0, i] = preprocessing_params["glucose_mean"]
            elif feature == "BloodPressure" and "bloodpressure_mean" in preprocessing_params:
                out[0, i] = preprocessing_params["bloodpressure_mean"]
            elif feature == "SkinThickness" and "skinthickness_median" in preprocessing_params:
                out[0, i] = preprocessing_params["skinthickness_median"]
            elif feature == "Insulin" and "insulin_median" in preprocessing_params:
                out[0, i] = preprocessing_params["insulin_median"]
            elif feature == "BMI" and "bmi_median" in preprocessing_params:
                out[0, i] = preprocessing_params["bmi_median"]
    return out


def _extract_shap_row(shap_values: Any) -> np.ndarray:
    """Normalize SHAP output to shape (n_features,) for one sample, positive class."""
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim == 2:
        arr = arr[0]
    return np.asarray(arr, dtype=float).reshape(-1)


def _feature_line_description(feature: str, feature_value: float) -> str:
    """Human-readable feature + value with units (thesis / report style)."""
    f = str(feature)
    v = float(feature_value)
    descriptions = {
        "Glucose": f"Glucose ({v:.0f} mg/dL)",
        "BloodPressure": f"Blood pressure ({v:.0f} mmHg)",
        "SkinThickness": f"Skin thickness ({v:.0f} mm)",
        "Insulin": f"Insulin ({v:.1f} μU/mL)",
        "BMI": f"BMI ({v:.1f} kg/m²)",
        "Age": f"Age ({v:.0f} years)",
        "Pregnancies": f"Number of pregnancies ({v:.0f})",
        "DPF": f"Diabetes pedigree function ({v:.3f})",
        "Gender": f"Gender ({'Male' if v == 1 else 'Female'})",
    }
    return descriptions.get(f, f"{f} ({v:.3g})")


def _human_explanation_text(
    prediction_label: str,
    probability: float,
    feature_contributions: Dict[str, Dict[str, Any]],
    top_feature_names: List[str],
) -> str:
    """Natural-language SHAP summary similar to documentation examples."""
    lines = [
        f"The model predicts: **{prediction_label}** (confidence: {probability:.1%})",
        "",
        "**Key factors:**",
    ]
    for i, feat in enumerate(top_feature_names[:5], 1):
        c = feature_contributions.get(feat)
        if not c:
            continue
        shap_val = float(c["shap_value"])
        fv = float(c["feature_value"])
        strength = (
            "strongly" if abs(shap_val) > 0.1 else "moderately" if abs(shap_val) > 0.05 else "slightly"
        )
        direction = "increased" if shap_val > 0 else "decreased"
        phrase = _feature_line_description(feat, fv)
        lines.append(f"{i}. **{phrase}**: {strength} {direction} your diabetes risk.")
    return "\n".join(lines)


def explain_sample_shap(
    model: Any,
    X_input: pd.DataFrame | np.ndarray | Sequence[float],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """
    SHAP explanation for a single row (used by prediction_pipeline).

    Returns keys including prediction label string, confidence, top_features as list of dicts.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. pip install shap")

    if isinstance(X_input, pd.DataFrame):
        X_df = X_input.copy()
        if list(X_df.columns) != list(feature_names):
            X_df = X_df[list(feature_names)]
    else:
        arr = np.asarray(X_input, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        X_df = pd.DataFrame(arr, columns=list(feature_names))

    pred = int(np.asarray(model.predict(X_df)).reshape(-1)[0])
    proba = np.asarray(model.predict_proba(X_df))[0]
    positive_idx = 1 if proba.shape[0] > 1 else 0
    probability = float(proba[positive_idx])
    prediction_label = "Diabetic" if pred == 1 else "Non-Diabetic"

    explainer = shap.TreeExplainer(model)
    sv = _extract_shap_row(explainer.shap_values(X_df))
    if sv.size != len(feature_names):
        raise ValueError("SHAP length does not match feature_names.")

    ranked_idx = np.argsort(np.abs(sv))[::-1][:5]
    top_features: List[Dict[str, str]] = []
    for j in ranked_idx:
        impact = float(sv[j])
        feat = str(feature_names[j])
        effect = "increases risk" if impact >= 0 else "decreases risk"
        top_features.append(
            {"feature": feat, "impact": f"{impact:+.4f}", "effect": effect}
        )

    explanation_text = _human_explanation_text(
        prediction_label,
        probability,
        {
            feature_names[k]: {
                "shap_value": float(sv[k]),
                "feature_value": float(X_df.iloc[0, k]),
            }
            for k in range(len(feature_names))
        },
        [feature_names[j] for j in ranked_idx],
    )

    return {
        "prediction": prediction_label,
        "prediction_int": pred,
        "confidence": probability,
        "top_features": top_features,
        "explanation_text": explanation_text,
    }


def explain_prediction(
    input_data: Any,
    model: Any | None = None,
    preprocessing_params: Dict[str, Any] | None = None,
    feature_names: List[str] | None = None,
    background_data: Any = None,
    explainer: Any = None,
    return_shap_values: bool = True,
) -> Dict[str, Any]:
    """
    Explain one patient (dict or array) using preprocessing + optional SHAP.

    Compatible with app_xai.py: returns prediction (int), prediction_label, probability,
    top_features (list of feature name strings), feature_contributions, explanation_text.
    """
    if model is None or preprocessing_params is None or feature_names is None:
        loaded_model, loaded_params, loaded_features = load_model_and_data()
        if model is None:
            model = loaded_model
        if preprocessing_params is None:
            preprocessing_params = loaded_params
        if feature_names is None:
            feature_names = loaded_features

    processed = preprocess_input(input_data, preprocessing_params, feature_names)
    pred = int(np.asarray(model.predict(processed)).reshape(-1)[0])
    proba = np.asarray(model.predict_proba(processed))[0]
    positive_idx = 1 if proba.shape[0] > 1 else 0
    probability = float(proba[positive_idx])
    prediction_label = "Diabetic" if pred == 1 else "Non-Diabetic"

    if not SHAP_AVAILABLE:
        return {
            "prediction": pred,
            "prediction_label": prediction_label,
            "probability": probability,
            "feature_contributions": {},
            "explanation_text": (
                f"The model predicts **{prediction_label}** (confidence: {probability:.1%}). "
                "Install SHAP for detailed feature explanations: pip install shap"
            ),
            "top_features": [],
        }

    X_df = pd.DataFrame(processed, columns=list(feature_names))

    if explainer is None:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_df)
    sv = _extract_shap_row(shap_values)

    feature_contributions: Dict[str, Dict[str, Any]] = {}
    for i, feat in enumerate(feature_names):
        if i < len(sv):
            feature_contributions[feat] = {
                "shap_value": float(sv[i]),
                "feature_value": float(processed[0, i]),
                "contribution": float(sv[i]),
            }

    sorted_feats = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]["shap_value"]),
        reverse=True,
    )
    top_feature_names = [name for name, _ in sorted_feats[:5]]

    explanation_text = _human_explanation_text(
        prediction_label, probability, feature_contributions, top_feature_names
    )

    result: Dict[str, Any] = {
        "prediction": pred,
        "prediction_label": prediction_label,
        "probability": probability,
        "feature_contributions": feature_contributions,
        "explanation_text": explanation_text,
        "top_features": top_feature_names,
    }
    if return_shap_values:
        result["shap_values"] = sv.tolist()
    return result


def plot_summary(model: Any, X_train: pd.DataFrame | np.ndarray, max_samples: int = 500) -> None:
    """Optional: SHAP beeswarm summary (requires matplotlib backend)."""
    if not SHAP_AVAILABLE:
        raise ImportError("pip install shap")
    X = X_train
    if hasattr(X, "iloc") and len(X) > max_samples:
        X = X.sample(min(max_samples, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1] if len(vals) > 1 else vals[0]
    shap.summary_plot(vals, X)


def plot_force(model: Any, X_input: pd.DataFrame | np.ndarray, feature_names: Sequence[str] | None = None) -> Any:
    """Optional: matplotlib force plot for one row."""
    if not SHAP_AVAILABLE:
        raise ImportError("pip install shap")
    if isinstance(X_input, pd.DataFrame):
        row = X_input.iloc[[0]]
    else:
        arr = np.asarray(X_input, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = list(feature_names) if feature_names is not None else None
        row = pd.DataFrame(arr, columns=cols) if cols else pd.DataFrame(arr)

    explainer = shap.TreeExplainer(model)
    sv = _extract_shap_row(explainer.shap_values(row))
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and np.asarray(ev).size > 1:
        ev = float(np.asarray(ev).reshape(-1)[1])
    else:
        ev = float(np.asarray(ev).reshape(-1)[0])

    import matplotlib.pyplot as plt

    shap.force_plot(ev, sv, row.iloc[0], matplotlib=True, feature_names=list(row.columns))
    return plt.gcf()


def plot_waterfall(model: Any, X_input: pd.DataFrame | np.ndarray, feature_names: Sequence[str]) -> None:
    """Optional: waterfall plot for one row."""
    if not SHAP_AVAILABLE:
        raise ImportError("pip install shap")
    row = (
        X_input.iloc[[0]]
        if hasattr(X_input, "iloc")
        else pd.DataFrame(np.asarray(X_input).reshape(1, -1), columns=list(feature_names))
    )
    explainer = shap.TreeExplainer(model)
    sv = _extract_shap_row(explainer.shap_values(row))
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and np.asarray(ev).size > 1:
        base = float(np.asarray(ev).reshape(-1)[1])
    else:
        base = float(np.asarray(ev).reshape(-1)[0])

    explanation = shap.Explanation(
        values=sv,
        base_values=base,
        data=row.iloc[0].values,
        feature_names=list(feature_names),
    )
    shap.plots.waterfall(explanation)
