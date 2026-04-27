"""
Flask Web Application with Explainable AI (XAI) Integration

This is an enhanced version of app.py that includes SHAP-based explanations
for diabetes predictions.

Author: AI Assistant
Date: 2024
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sys
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

# Try to import XAI module (SHAP lives inside xai; import may fail without deps)
explain_prediction = None  # type: ignore
preprocess_input = None  # type: ignore
load_model_and_data = None  # type: ignore
try:
    from xai import explain_prediction as _explain_prediction
    from xai import load_model_and_data as _load_model_and_data
    from xai import preprocess_input as _preprocess_input

    explain_prediction = _explain_prediction
    load_model_and_data = _load_model_and_data
    preprocess_input = _preprocess_input
    XAI_AVAILABLE = True
except ImportError as _xai_import_err:
    XAI_AVAILABLE = False
    print(f"⚠ XAI module not available ({_xai_import_err}). Install deps: pip install shap pandas numpy")

# Try to import Recommendation module
try:
    from recommendation import generate_recommendation, generate_human_friendly_summary
    RECOMMENDATION_AVAILABLE = True
except ImportError:
    RECOMMENDATION_AVAILABLE = False
    print("⚠ Recommendation module not available.")

# Legacy model loading code (for backward compatibility)
import sklearn.ensemble
import sklearn.tree
import types

if not hasattr(sklearn.ensemble, 'forest'):
    sklearn.ensemble.forest = types.ModuleType('forest')
    sklearn.ensemble.forest.__dict__.update(sklearn.ensemble.__dict__)
    sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble.forest

if not hasattr(sklearn.tree, 'tree'):
    sklearn.tree.tree = types.ModuleType('tree')
    sklearn.tree.tree.__dict__.update(sklearn.tree.__dict__)
    sys.modules['sklearn.tree.tree'] = sklearn.tree.tree

import sklearn.tree._tree as _tree_module
_original_tree_setstate = _tree_module.Tree.__setstate__

def _patched_tree_setstate(self, state):
    """Wrapper that patches old tree format before calling original __setstate__"""
    if isinstance(state, dict) and 'tree_' in state:
        tree_state = state['tree_']
        if hasattr(tree_state, 'dtype') and tree_state.dtype.names:
            dtype_names = tree_state.dtype.names
            if 'missing_go_to_left' not in dtype_names:
                new_dtype_list = []
                for name in dtype_names:
                    new_dtype_list.append((name, tree_state.dtype[name]))
                new_dtype_list.append(('missing_go_to_left', 'u1'))
                new_dtype = np.dtype(new_dtype_list)
                new_tree = np.zeros(tree_state.shape, dtype=new_dtype)
                for name in dtype_names:
                    new_tree[name] = tree_state[name]
                new_tree['missing_go_to_left'] = 0
                state['tree_'] = new_tree
    return _original_tree_setstate(self, state)

try:
    setattr(_tree_module.Tree, '__setstate__', _patched_tree_setstate)
except (TypeError, AttributeError):
    pass

class OldSklearnUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        if pid[0] == 'sklearn.ensemble.forest':
            return sklearn.ensemble
        elif pid[0] == 'sklearn.tree.tree':
            return sklearn.tree
        else:
            return super().persistent_load(pid)

# ============================================================================
# MODEL LOADING
# ============================================================================

import joblib

# Try to load new XAI model first, fall back to legacy model
classifier = None
preprocessing_params = None
feature_names = None
load_error = None

# Try loading new XAI model
new_model_filename = 'diabetes-prediction-xai-model.pkl'
new_preprocessing_filename = 'preprocessing_params.pkl'
new_feature_names_filename = 'feature_names.pkl'

if os.path.exists(new_model_filename):
    try:
        classifier = joblib.load(new_model_filename)
        preprocessing_params = joblib.load(new_preprocessing_filename)
        feature_names = joblib.load(new_feature_names_filename)
        print("✓ Loaded new XAI model")
    except Exception as e:
        print(f"⚠ Could not load new XAI model: {e}")
        print("  Falling back to legacy model...")

# Fall back to legacy model if new model not available
if classifier is None:
    legacy_model_filename = 'diabetes-prediction-rfc-model.pkl'
    legacy_preprocessing_filename = 'preprocessing_params.pkl'
    
    for loader_name, loader_func in [
        ('joblib', lambda: joblib.load(legacy_model_filename)),
        ('custom unpickler', lambda: OldSklearnUnpickler(open(legacy_model_filename, 'rb')).load()),
        ('standard pickle', lambda: pickle.load(open(legacy_model_filename, 'rb')))
    ]:
        try:
            classifier = loader_func()
            break
        except Exception as e:
            load_error = e
            continue
    
    try:
        preprocessing_params = joblib.load(legacy_preprocessing_filename)
    except:
        preprocessing_params = {
            'glucose_mean': 121.0,
            'bloodpressure_mean': 72.0,
            'skinthickness_median': 29.0,
            'insulin_median': 102.5,
            'bmi_median': 32.0
        }
    
    # Default feature names for legacy model
    if feature_names is None:
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DPF', 'Age']

if classifier is None:
    print("\n" + "="*70)
    print("ERROR: Failed to load model!")
    print("="*70)
    print(f"\nError details: {load_error}\n")
    print("SOLUTION: Train the model first.")
    print("\nTo train the model, run:")
    print("  python model.py")
    print("="*70)
    raise RuntimeError("Model loading failed. Please train the model using model.py")

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)


def _explanation_for_template(raw: dict) -> dict:
    """
    Build the `explanation` object expected by templates/result_xai.html.

    Uses plain dicts so Jinja can access explanation.probability, etc.
    """
    fc = raw.get("feature_contributions") or {}
    top = raw.get("top_features") or []
    if top and isinstance(top[0], dict):
        top = [item.get("feature", str(item)) for item in top]
    return {
        "prediction": int(raw.get("prediction", 0)),
        "prediction_label": str(raw.get("prediction_label", "")),
        "probability": float(raw.get("probability", 0.0)),
        "explanation_text": str(raw.get("explanation_text", "")),
        "top_features": list(top),
        "feature_contributions": dict(fc),
    }


def _preprocess_row(input_data: dict, preprocessing_params: dict, feature_names: list) -> np.ndarray:
    """Match training-time imputation; uses xai.preprocess_input when available."""
    if preprocess_input is not None:
        return preprocess_input(input_data, preprocessing_params, feature_names)
    row = [input_data.get(f, 0) for f in feature_names]
    arr = np.array([row], dtype=float)
    out = arr.copy()
    for i, feature in enumerate(feature_names):
        if i >= out.shape[1]:
            break
        if float(out[0, i]) == 0.0:
            if feature == 'Glucose':
                out[0, i] = preprocessing_params.get('glucose_mean', 121.0)
            elif feature == 'BloodPressure':
                out[0, i] = preprocessing_params.get('bloodpressure_mean', 72.0)
            elif feature == 'SkinThickness':
                out[0, i] = preprocessing_params.get('skinthickness_median', 29.0)
            elif feature == 'Insulin':
                out[0, i] = preprocessing_params.get('insulin_median', 102.5)
            elif feature == 'BMI':
                out[0, i] = preprocessing_params.get('bmi_median', 32.0)
    return out


def _minimal_recommendation(message: str) -> dict:
    """Non-empty payload so disclaimer / tips sections can render."""
    return {
        "risk_bucket": "Unknown",
        "probability": 0.0,
        "priority_interventions": [],
        "diet_plan": None,
        "exercise_plan": None,
        "nudges": [message] if message else [],
        "disclaimer": (
            "This is an educational helper only. It is NOT medical advice. "
            "Consult a qualified clinician for diagnosis and treatment."
        ),
    }


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction: always render result_xai.html with explanation + recommendation."""
    if request.method == 'POST':
        try:
            # Get form data
            preg = int(request.form.get('pregnancies', 0))
            glucose = int(request.form.get('glucose', 0))
            bp = int(request.form.get('bloodpressure', 0))
            st = int(request.form.get('skinthickness', 0))
            insulin = int(request.form.get('insulin', 0))
            bmi = float(request.form.get('bmi', 0))
            dpf = float(request.form.get('dpf', 0))
            age = int(request.form.get('age', 0))
            
            # Handle Gender if present in form
            gender = request.form.get('gender', None)
            
            # Get user preferences for recommendations
            diet_pref = request.form.get('diet_pref', 'non-veg')
            activity_level = request.form.get('activity_level', 'sedentary')
            time_per_day = int(request.form.get('time_per_day_min', 30))
            
            # Prepare input data dictionary
            input_data = {
                'Pregnancies': preg,
                'Glucose': glucose,
                'BloodPressure': bp,
                'SkinThickness': st,
                'Insulin': insulin,
                'BMI': bmi,
                'DPF': dpf,
                'Age': age
            }
            
            # Add Gender if provided
            if gender is not None and gender != '':
                input_data['Gender'] = 1 if gender.lower() in ['male', 'm', '1'] else 0
            
            preferences = {
                'diet_pref': diet_pref,
                'activity_level': activity_level,
                'time_per_day_min': time_per_day,
            }

            # Baseline prediction (always) — same preprocessing as explain_prediction
            processed = _preprocess_row(input_data, preprocessing_params, feature_names)

            pred_arr = classifier.predict(processed)
            pred_int = int(np.asarray(pred_arr).reshape(-1)[0])
            proba_row = np.asarray(classifier.predict_proba(processed))[0]
            positive_idx = 1 if proba_row.shape[0] > 1 else 0
            probability = float(proba_row[positive_idx])
            pred_label = 'Diabetic' if pred_int == 1 else 'Non-Diabetic'

            explanation_raw: dict
            if XAI_AVAILABLE and explain_prediction is not None:
                try:
                    explanation_raw = explain_prediction(
                        input_data,
                        model=classifier,
                        preprocessing_params=preprocessing_params,
                        feature_names=feature_names,
                    )
                except Exception as e:
                    print(f"⚠ XAI explanation failed: {e}")
                    explanation_raw = {
                        'prediction': pred_int,
                        'prediction_label': pred_label,
                        'probability': probability,
                        'explanation_text': (
                            f"The model predicts **{pred_label}** (confidence: {probability:.1%}).\n\n"
                            f"SHAP explanation could not be generated: {e}"
                        ),
                        'top_features': [],
                        'feature_contributions': {},
                    }
            else:
                explanation_raw = {
                    'prediction': pred_int,
                    'prediction_label': pred_label,
                    'probability': probability,
                    'explanation_text': (
                        f"The model predicts **{pred_label}** (confidence: {probability:.1%}).\n\n"
                        "Install the `xai` dependencies (e.g. `pip install shap`) for SHAP-based explanations."
                    ),
                    'top_features': [],
                    'feature_contributions': {},
                }

            explanation_payload = _explanation_for_template(explanation_raw)

            recommendation_payload = None
            if RECOMMENDATION_AVAILABLE:
                try:
                    recommendation_payload = generate_recommendation(
                        input_data,
                        explanation_raw,
                        preferences,
                    )
                except Exception as e:
                    print(f"⚠ Recommendation generation failed: {e}")
                    recommendation_payload = _minimal_recommendation(
                        f"Recommendations could not be generated: {e}"
                    )
            else:
                recommendation_payload = _minimal_recommendation(
                    "Recommendation module is not available."
                )

            return render_template(
                'result_xai.html',
                prediction=pred_int,
                explanation=explanation_payload,
                recommendation=recommendation_payload,
            )
            
        except Exception as e:
            return render_template('error.html', error_message=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions with JSON response."""
    try:
        data = request.get_json()
        
        # Prepare input data
        input_data = {
            'Pregnancies': int(data.get('pregnancies', 0)),
            'Glucose': int(data.get('glucose', 0)),
            'BloodPressure': int(data.get('bloodpressure', 0)),
            'SkinThickness': int(data.get('skinthickness', 0)),
            'Insulin': int(data.get('insulin', 0)),
            'BMI': float(data.get('bmi', 0)),
            'DPF': float(data.get('dpf', 0)),
            'Age': int(data.get('age', 0))
        }
        
        if 'gender' in data:
            input_data['Gender'] = 1 if str(data['gender']).lower() in ['male', 'm', '1'] else 0
        
        # Get preferences from request
        preferences = {
            'diet_pref': data.get('diet_pref', 'non-veg'),
            'activity_level': data.get('activity_level', 'sedentary'),
            'time_per_day_min': int(data.get('time_per_day_min', 30))
        }
        
        # Generate explanation if XAI available
        if XAI_AVAILABLE:
            explanation = explain_prediction(
                input_data,
                model=classifier,
                preprocessing_params=preprocessing_params,
                feature_names=feature_names
            )
            
            response_data = {
                'success': True,
                'prediction': explanation['prediction'],
                'prediction_label': explanation['prediction_label'],
                'probability': explanation['probability'],
                'explanation_text': explanation['explanation_text'],
                'top_features': explanation['top_features'],
                'feature_contributions': explanation['feature_contributions']
            }
            
            # Add recommendations if available
            if RECOMMENDATION_AVAILABLE:
                try:
                    recommendation = generate_recommendation(
                        input_data,
                        explanation,
                        preferences
                    )
                    response_data['recommendation'] = recommendation
                    response_data['recommendation_summary'] = generate_human_friendly_summary(recommendation)
                except Exception as e:
                    response_data['recommendation_error'] = str(e)
            
            return jsonify(response_data)
        else:
            # Basic prediction
            processed_input = _preprocess_row(input_data, preprocessing_params, feature_names)
            prediction = classifier.predict(processed_input)[0]
            probability = classifier.predict_proba(processed_input)[0][1]
            
            return jsonify({
                'success': True,
                'prediction': int(prediction),
                'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
                'probability': float(probability)
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DIABETES PREDICTION WEB APP WITH XAI & RECOMMENDATIONS")
    print("="*70)
    if XAI_AVAILABLE:
        print("✓ XAI explanations enabled")
    else:
        print("⚠ XAI explanations disabled (SHAP not installed)")
    if RECOMMENDATION_AVAILABLE:
        print("✓ Recommendation engine enabled")
    else:
        print("⚠ Recommendation engine disabled")
    print("="*70)
    app.run(debug=True)

