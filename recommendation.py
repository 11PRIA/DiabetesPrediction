"""Rule-based recommendation engine driven by prediction + SHAP outputs."""

from __future__ import annotations

from typing import Any, Dict, List


LOW_RISK_MAX = 0.4
MODERATE_RISK_MAX = 0.7

DIET_MAP = {
    "veg": {
        "breakfast": "Oatmeal with berries and nuts, or whole grain toast with avocado",
        "lunch": "Dal + mixed vegetable sabzi + 1–2 whole wheat rotis (portion-controlled)",
        "dinner": "Grilled vegetables with tofu and brown rice, or chickpea curry with roti",
    },
    "non-veg": {
        "breakfast": "Scrambled eggs with whole grain toast and vegetables (2 eggs, 2 slices, 1 cup veg)",
        "lunch": "Grilled chicken salad with olive oil dressing (4 oz chicken, 2 cups greens)",
        "dinner": "Baked salmon with sweet potato and broccoli (4 oz salmon, 1 medium potato, 1 cup veg)",
    },
}


def _risk_bucket(probability: float) -> str:
    if probability < LOW_RISK_MAX:
        return "Low"
    if probability <= MODERATE_RISK_MAX:
        return "Moderate"
    return "High"


def _normalize_profile(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = user_profile or {}
    diet_type = str(profile.get("diet_type", "non-veg")).lower()
    if diet_type not in {"veg", "non-veg"}:
        diet_type = "non-veg"

    activity_level = str(profile.get("activity_level", "medium")).lower()
    if activity_level not in {"low", "medium", "high"}:
        activity_level = "medium"

    try:
        time_available = int(profile.get("time_available", 30))
    except (TypeError, ValueError):
        time_available = 30
    time_available = max(10, min(time_available, 180))

    return {
        "diet_type": diet_type,
        "activity_level": activity_level,
        "time_available": time_available,
    }


def _extract_feature_names(shap_output: Any) -> List[str]:
    if isinstance(shap_output, dict):
        top_features = shap_output.get("top_features", [])
    elif isinstance(shap_output, list):
        top_features = shap_output
    else:
        top_features = []

    names: List[str] = []
    for item in top_features:
        if isinstance(item, dict):
            name = str(item.get("feature", "")).strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name.lower())
    return names


def _priority_interventions(risk_bucket: str, feature_names: List[str]) -> List[str]:
    actions: List[str] = []
    feature_set = set(feature_names)

    if "glucose" in feature_set:
        actions.extend(
            [
                "Reduce refined carbohydrates and sugary drinks",
                "Favor low-glycemic foods and balanced meal timing",
            ]
        )
    if "bmi" in feature_set:
        actions.extend(
            [
                "Target gradual fat loss with a calorie deficit",
                "Add at least 2 weekly resistance-training sessions",
            ]
        )
    if "bloodpressure" in feature_set:
        actions.append("Lower sodium intake and prioritize home-cooked meals")
    if "insulin" in feature_set:
        actions.append("Prefer high-fiber meals to improve insulin sensitivity")
    if "age" in feature_set:
        actions.append("Use sustainable low-impact activity and recovery routines")

    if risk_bucket == "High":
        actions.insert(0, "Book a physician consultation for clinical follow-up")
    elif risk_bucket == "Moderate":
        actions.insert(0, "Review fasting glucose and HbA1c with a clinician")
    else:
        actions.insert(0, "Maintain preventive habits and monitor quarterly")

    deduped: List[str] = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)
    return deduped[:6]


def _diet_plan(risk_bucket: str, diet_type: str, feature_names: List[str]) -> Dict[str, Any]:
    base = DIET_MAP[diet_type]
    snacks = (
        "Apple with almond butter, or carrot sticks with hummus (veg)"
        if diet_type == "veg"
        else "Greek yogurt with nuts (1 cup yogurt, 1 oz nuts)"
    )
    weekly_focus = [
        "Keep total added sugar minimal across the week",
        "Prioritize vegetables and protein in every main meal",
    ]
    if "glucose" in feature_names:
        weekly_focus.append("Swap high-GI carbs for low-GI options")
    if "bmi" in feature_names:
        weekly_focus.append("Use portion control in dinner meals")
    if risk_bucket == "High":
        weekly_focus.append("Avoid late-night snacking and sweetened beverages")
    if "bloodpressure" in feature_names:
        weekly_focus.append("Reduce sodium intake to <2300mg/day")
        weekly_focus.append("Limit processed foods")

    risk_phrase = f"{risk_bucket.lower()} diabetes risk"
    diet_label = "non-veg" if diet_type == "non-veg" else "vegetarian"
    return {
        "summary": (
            f"Based on your {risk_phrase}, here's a personalized {diet_label} diet plan. "
            "Emphasize whole foods, fiber, and lean protein; limit refined carbs and sugary drinks."
        ),
        "diet_type": diet_label,
        "daily_guidelines": {
            "breakfast": base["breakfast"],
            "lunch": base["lunch"],
            "dinner": base["dinner"],
            "snacks": snacks,
            "hydration": "2-3 liters water/day, limit sweetened beverages",
        },
        "weekly_focus": weekly_focus,
        "general_guidelines": [
            "Prefer whole foods over packaged snacks",
            "Balance each meal with protein + fiber",
        ],
    }


def _exercise_plan(
    risk_bucket: str, activity_level: str, time_available: int, feature_names: List[str]
) -> Dict[str, Any]:
    base_minutes = {"low": 90, "medium": 150, "high": 210}[activity_level]
    if risk_bucket == "High":
        base_minutes = max(base_minutes, 180)
    if risk_bucket == "Low":
        base_minutes = max(base_minutes, 120)

    sessions = max(3, min(6, base_minutes // max(time_available, 20)))
    daily_steps_target = 8000 if risk_bucket in {"Moderate", "High"} else 7000

    weekly_plan = [
        {"day": "Monday", "activity": f"{time_available} min brisk walk", "type": "cardio"},
        {"day": "Wednesday", "activity": f"{time_available} min cycling/walk", "type": "cardio"},
        {"day": "Friday", "activity": "25 min strength (bodyweight)", "type": "strength"},
        {"day": "Sunday", "activity": "Mobility + stretching 20 min", "type": "recovery"},
    ]
    if "bmi" in feature_names:
        weekly_plan.append({"day": "Tuesday", "activity": "20 min resistance training", "type": "strength"})

    strength_sessions = 2 if "bmi" in feature_names or risk_bucket == "High" else 1

    return {
        "summary": f"{sessions} sessions/week targeting {base_minutes} minutes total activity.",
        "weekly_plan": weekly_plan,
        "goals": {
            "weekly_minutes": int(base_minutes),
            "daily_steps_target": int(daily_steps_target),
            "strength_sessions_per_week": int(strength_sessions),
        },
    }


def _behavioral_nudges(risk_bucket: str, feature_names: List[str]) -> List[str]:
    nudges = [
        "Track body weight once weekly (same day/time).",
        "Sleep 7-8 hours to support glucose control.",
        "Set one small weekly habit goal and review on Sunday.",
    ]
    if "glucose" in feature_names:
        nudges.append("Maintain a simple post-meal glucose log if advised by clinician.")
    if risk_bucket == "High":
        nudges.append("Share progress with an accountability partner each week.")
    return nudges[:6]


def generate_recommendations(
    prediction: str,
    probability: float,
    shap_output: Dict[str, Any] | List[Dict[str, Any]],
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create personalized recommendations using risk + SHAP feature drivers.

    Args:
        prediction: 'Diabetic' or 'Non-Diabetic'.
        probability: Model probability for diabetic class.
        shap_output: SHAP explanation payload containing top contributing features.
        user_profile: Dict with diet_type, activity_level, time_available.

    Returns:
        Dictionary with interventions, diet plan, exercise plan, and nudges.
    """
    try:
        probability_val = float(probability)
    except (TypeError, ValueError):
        probability_val = 0.0
    probability_val = max(0.0, min(1.0, probability_val))

    bucket = _risk_bucket(probability_val)
    profile = _normalize_profile(user_profile)
    feature_names = _extract_feature_names(shap_output)

    priority = _priority_interventions(bucket, feature_names)
    diet_plan = _diet_plan(bucket, profile["diet_type"], feature_names)
    exercise_plan = _exercise_plan(
        bucket, profile["activity_level"], profile["time_available"], feature_names
    )
    behavioral_nudges = _behavioral_nudges(bucket, feature_names)

    disclaimer = (
        "This is an educational helper only. It is NOT medical advice. "
        "Consult a qualified clinician for diagnosis and treatment."
    )

    out = {
        "risk_bucket": bucket,
        "probability": probability_val,
        "prediction": prediction,
        "prediction_label": prediction if isinstance(prediction, str) else str(prediction),
        "priority_interventions": priority,
        "diet_plan": diet_plan,
        "exercise_plan": exercise_plan,
        "behavioral_nudges": behavioral_nudges,
        "nudges": behavioral_nudges,
        "disclaimer": disclaimer,
    }
    return out


DISCLAIMER_TEXT = (
    "This is an educational helper only. It is NOT medical advice. "
    "Consult a qualified clinician for diagnosis and treatment."
)


def _map_flask_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Map app_xai form keys (diet_pref, activity_level, time_per_day_min) to engine keys."""
    p = preferences or {}
    diet_raw = str(p.get("diet_pref", p.get("diet_type", "non-veg"))).lower()
    if diet_raw in ("vegetarian", "veg", "vegan"):
        diet_type = "veg"
    else:
        diet_type = "non-veg"

    act = str(p.get("activity_level", "medium")).lower()
    activity_map = {
        "sedentary": "low",
        "light": "low",
        "moderate": "medium",
        "active": "high",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    activity_level = activity_map.get(act, "medium")

    try:
        t = int(p.get("time_per_day_min", p.get("time_available", 30)))
    except (TypeError, ValueError):
        t = 30

    return {"diet_type": diet_type, "activity_level": activity_level, "time_available": t}


def _xai_summary_for_engine(xai_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure SHAP payload has top_features understood by generate_recommendations."""
    if not xai_summary:
        return {}
    out = dict(xai_summary)
    # Engine expects list of dicts with 'feature' for top_features; strings also work via _extract_feature_names
    if "top_features" in out and out["top_features"] and isinstance(out["top_features"][0], str):
        feats = out["top_features"]
        fc = out.get("feature_contributions") or {}
        out["top_features"] = [
            {"feature": name, "shap_value": float(fc.get(name, {}).get("shap_value", 0.0))}
            for name in feats
        ]
    return out


def generate_recommendation(
    input_dict: Dict[str, Any],
    xai_summary: Dict[str, Any],
    preferences: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Flask-compatible wrapper: same signature as legacy recommendation module.

    Uses prediction label string and probability from explain_prediction().
    """
    pred_label = xai_summary.get("prediction_label")
    if pred_label is None:
        pred_label = "Diabetic" if int(xai_summary.get("prediction", 0)) == 1 else "Non-Diabetic"
    prob = float(xai_summary.get("probability", 0.0))
    profile = _map_flask_preferences(preferences)
    payload = _xai_summary_for_engine(xai_summary)
    result = generate_recommendations(
        prediction=str(pred_label),
        probability=prob,
        shap_output=payload,
        user_profile=profile,
    )
    result.setdefault("disclaimer", DISCLAIMER_TEXT)
    result.setdefault("nudges", result.get("behavioral_nudges", []))
    return result


def generate_human_friendly_summary(recommendation: Dict[str, Any]) -> str:
    """Short natural-language summary for API consumers."""
    bucket = recommendation.get("risk_bucket", "")
    prob = float(recommendation.get("probability", 0.0))
    parts = [
        f"Risk level: {bucket} (estimated probability {prob:.1%}).",
        "",
        "Priority actions:",
    ]
    for i, a in enumerate(recommendation.get("priority_interventions", [])[:5], 1):
        parts.append(f"{i}. {a}")
    dp = recommendation.get("diet_plan", {})
    if dp.get("summary"):
        parts.extend(["", f"Diet: {dp['summary']}"])
    ep = recommendation.get("exercise_plan", {})
    if ep.get("summary"):
        parts.extend(["", f"Activity: {ep['summary']}"])
    parts.extend(["", f"Note: {recommendation.get('disclaimer', DISCLAIMER_TEXT)}"])
    return "\n".join(parts)
