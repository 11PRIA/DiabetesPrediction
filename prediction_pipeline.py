"""End-to-end prediction pipeline + optional FastAPI endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from recommendation import generate_recommendations
from xai import explain_sample_shap

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def _to_single_row_df(input_data: Dict[str, Any], feature_names: Sequence[str]) -> pd.DataFrame:
    """Build a one-row DataFrame in strict feature order."""
    if not isinstance(input_data, dict):
        raise ValueError("input_data must be a dictionary.")

    missing = [name for name in feature_names if name not in input_data]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    row = {name: input_data[name] for name in feature_names}
    return pd.DataFrame([row], columns=list(feature_names))


def predict_and_explain(
    model: Any,
    input_data: Dict[str, Any],
    feature_names: Sequence[str],
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict diabetes risk, explain via SHAP, and generate recommendations.

    Args:
        model: Trained classifier with predict/predict_proba.
        input_data: Single patient data dictionary.
        feature_names: Ordered model feature names.
        user_profile: User context for recommendation personalization.

    Returns:
        Combined dictionary with prediction, explanation, and recommendations.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must implement predict_proba().")
    if not feature_names:
        raise ValueError("feature_names cannot be empty.")

    X_input = _to_single_row_df(input_data, feature_names)

    proba = float(np.asarray(model.predict_proba(X_input))[0][1])
    xai_output = explain_sample_shap(model=model, X_input=X_input, feature_names=feature_names)

    recommendations = generate_recommendations(
        prediction=xai_output["prediction"],
        probability=proba,
        shap_output=xai_output,
        user_profile=user_profile,
    )

    return {
        "prediction": xai_output["prediction"],
        "confidence": float(proba),
        "explanation": {
            "top_features": xai_output["top_features"],
            "explanation_text": xai_output["explanation_text"],
        },
        "recommendations": recommendations,
    }


if FASTAPI_AVAILABLE:
    app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

    class UserProfile(BaseModel):
        diet_type: str = Field(default="non-veg", examples=["veg", "non-veg"])
        activity_level: str = Field(default="medium", examples=["low", "medium", "high"])
        time_available: int = Field(default=30, ge=10, le=180)

    class PredictRequest(BaseModel):
        input_data: Dict[str, float]
        user_profile: UserProfile = UserProfile()

    # Set these once from your startup code.
    MODEL: Any = None
    FEATURE_NAMES: List[str] = []

    @app.post("/predict")
    def predict_endpoint(payload: PredictRequest) -> Dict[str, Any]:
        """
        POST /predict
        Body:
        {
          "input_data": {...all model features...},
          "user_profile": {"diet_type":"veg","activity_level":"medium","time_available":30}
        }
        """
        if MODEL is None or not FEATURE_NAMES:
            raise HTTPException(
                status_code=500,
                detail="MODEL/FEATURE_NAMES are not configured. Set them before serving requests.",
            )

        try:
            return predict_and_explain(
                model=MODEL,
                input_data=payload.input_data,
                feature_names=FEATURE_NAMES,
                user_profile=payload.user_profile.model_dump(),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
