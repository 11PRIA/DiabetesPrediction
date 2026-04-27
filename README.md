# Diabetes Prediction, Explanation, and Recommendation System

Full-stack Flask app for diabetes risk prediction with explainable AI (SHAP) and personalized lifestyle recommendations.

## What it does
- Predicts diabetes risk from clinical inputs (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age, optional Gender).
- Explains each prediction with SHAP: top features, per-feature contributions, human-readable text, optional plots.
- Generates personalized diet, exercise, and behavioral nudges based on risk level, SHAP drivers, and user preferences (diet type, activity level, time available).
- Provides both web UI and JSON API.

## Tech stack
- Flask for web/API.
- Models: scikit-learn RandomForest (default) or XGBoost (optional).
- Explainability: SHAP (TreeExplainer), optional LIME (mentioned in docs).
- Data/IO: pandas, numpy, joblib; Jinja2 templates for UI.

## Repository layout (key files)
```
app_xai.py                 # Main Flask app with XAI + recommendations
app.py                     # Legacy Flask app (prediction only)
xai.py                     # SHAP/LIME explainability helpers
recommendation.py          # Diet/exercise/nudge generator
model.py                   # Training script with XAI support (RF/XGBoost)
retrain_model.py           # Legacy training script (RF only)
example_usage.py           # CLI examples for XAI
templates/                 # index.html, result.html, result_xai.html
static/                    # static assets
diabetes-prediction-xai-model.pkl   # Trained XAI model (generated)
diabetes-prediction-rfc-model.pkl   # Legacy model (generated)
preprocessing_params.pkl           # Imputation stats (generated)
feature_names.pkl                  # Feature order (generated)
xai_visualizations/        # Saved SHAP plots (generated)
```

## End-to-end flow
1) User submits form (`/predict`) or JSON (`/api/predict`).
2) Inputs are sanitized and zeros imputed using `preprocessing_params.pkl` (means/medians).
3) Model predicts diabetes probability.
4) If SHAP available, `xai.explain_prediction` returns prediction label, probability, top features, and per-feature SHAP values; optional plots can be generated to `xai_visualizations/`.
5) `recommendation.generate_recommendation` uses probability + SHAP drivers + user preferences to build a diet plan, exercise plan, and nudges; `generate_human_friendly_summary` returns a concise text summary.
6) Response is rendered in `result_xai.html` (or `result.html` if XAI is unavailable) or returned as JSON from `/api/predict`.

## Model training (with XAI)
- File: `model.py`
- Data: `kaggle_diabetes.csv` (rename DiabetesPedigreeFunction → DPF; optional Gender auto-encoded).
- Preprocessing: replace 0s with NaN for Glucose, BloodPressure, SkinThickness, Insulin, BMI; impute (means/medians); store stats in `preprocessing_params.pkl`.
- Split: stratified train/test (default 80/20, random_state=42).
- Algorithms: RandomForest (default) or XGBoost (if installed).
- Saves: `diabetes-prediction-xai-model.pkl`, `preprocessing_params.pkl`, `feature_names.pkl`.
- Run: `python model.py`

## Legacy training (no XAI)
- File: `retrain_model.py`
- Trains RandomForest, saves `diabetes-prediction-rfc-model.pkl` + `preprocessing_params.pkl`.
- Used automatically if the XAI model is missing.

## Running the app
### Install deps
```
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
### Start (XAI version)
```
python app_xai.py
```
Open http://localhost:5000.

### API example
```
POST /api/predict
{
  "pregnancies": 2, "glucose": 130, "bloodpressure": 80,
  "skinthickness": 25, "insulin": 90, "bmi": 31.2,
  "dpf": 0.5, "age": 45,
  "diet_pref": "non-veg", "activity_level": "moderate", "time_per_day_min": 30
}
```
Response includes prediction, probability, SHAP drivers, and recommendations (if modules available).

## Recommendation engine (how it works)
- Risk buckets: Low (<0.2), Moderate (<0.5), High (≥0.5) based on probability.
- Maps top SHAP features to intervention themes (e.g., glucose → carb moderation; BMI/Weight → weight loss + cardio/strength; bloodpressure → sodium reduction).
- Builds diet plan from templates (veg/non-veg/vegan), weekly focus items, and general guidelines.
- Builds exercise plan adapted to activity level, available minutes/day, age, and weight/BMI focus; sets goals (weekly minutes, steps, strength sessions).
- Adds behavioral nudges (tracking, step goals, small-weekly-goal prompts).

## Explainability (SHAP)
- `xai.explain_prediction` wraps preprocessing + model inference + SHAP, returning:
  - `prediction`, `prediction_label`, `probability`
  - `top_features`, `feature_contributions` (SHAP values)
  - `explanation_text` (human-friendly)
- Visualization helpers: summary plot, force plot, waterfall, global feature importance (saved to `xai_visualizations/`).
- If SHAP is not installed, the app logs a warning and falls back to basic prediction.

## Datasets and required columns
- Required: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF (DiabetesPedigreeFunction), Age, Outcome.
- Optional: Gender (auto-encoded if present).
- File: CSV with headers; default name `kaggle_diabetes.csv`.

## Configuration knobs
- `model.py`: DATASET_PATH, MODEL_TYPE (random_forest|xgboost), RF/XGB hyperparameters, test size, random_state.
- `app_xai.py`: toggles based on availability of `shap` and `recommendation` modules; ports can be changed via `app.run(..., port=XXXX)`.

## Troubleshooting
- Model not found or incompatible: run `python model.py` (or `retrain_model.py` for legacy).
- SHAP import error: `pip install shap`.
- Dataset missing: place `kaggle_diabetes.csv` in project root or update DATASET_PATH.
- Port in use: change port in `app_xai.py` when calling `app.run`.

## Security & scope note
- This is an educational helper, not medical advice. Does not handle allergies, medications, or individual medical history. Always consult healthcare professionals.

## License
Educational use only.
