# Explainable AI (XAI) System for Diabetes Prediction

This document provides comprehensive instructions for using the Explainable AI (XAI) system integrated into your diabetes prediction project.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [File Structure](#file-structure)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Visualizations](#visualizations)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The XAI system enhances your diabetes prediction model with:

- **SHAP (SHapley Additive exPlanations)** for feature importance and per-patient explanations
- **LIME (optional)** for local explanation comparison
- **Human-friendly text explanations** for each prediction
- **Multiple visualization types**: summary plots, force plots, waterfall plots, and feature importance
- **Web integration** for displaying explanations in your Flask app

## 📦 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `shap>=0.42.0` - For SHAP explanations
- `lime>=0.2.0.1` - For LIME explanations (optional)
- All other required dependencies

### Step 2: Verify Installation

```bash
python -c "import shap; print('SHAP version:', shap.__version__)"
```

## 🚀 Quick Start

### 1. Train the Model with XAI Support

```bash
python model.py
```

This will:
- Load your dataset (`kaggle_diabetes.csv` by default)
- Train a Random Forest or XGBoost model
- Save the model and preprocessing parameters
- Support datasets with or without Gender feature

**Configuration**: Edit the configuration section in `model.py`:
```python
DATASET_PATH = 'kaggle_diabetes.csv'  # Your dataset path
MODEL_TYPE = 'random_forest'  # or 'xgboost'
```

### 2. Test the XAI System

```bash
python example_usage.py
```

This demonstrates:
- Basic prediction with explanation
- Multiple predictions
- Visualization generation
- Detailed feature analysis

### 3. Run the Web App with XAI

```bash
python app_xai.py
```

Then open `http://localhost:5000` in your browser.

## 📁 File Structure

```
Diabetes-Prediction/
├── model.py                    # Enhanced training script with XAI support
├── xai.py                      # XAI explanation module (SHAP/LIME)
├── app_xai.py                  # Flask app with XAI integration
├── example_usage.py            # Example usage demonstrations
├── requirements.txt            # Updated with XAI dependencies
├── XAI_README.md              # This file
│
├── diabetes-prediction-xai-model.pkl    # Trained model (generated)
├── preprocessing_params.pkl            # Preprocessing parameters (generated)
├── feature_names.pkl                   # Feature names (generated)
│
├── xai_visualizations/         # Generated visualization plots
│   ├── shap_summary_plot.png
│   ├── shap_force_plot.png
│   ├── shap_waterfall_plot.png
│   └── global_feature_importance.png
│
└── templates/
    ├── index.html              # Input form
    ├── result.html             # Basic result page
    └── result_xai.html         # Result page with XAI explanations
```

## 📖 Usage Guide

### Using the `explain_prediction()` Function

The main function for generating explanations:

```python
from xai import explain_prediction

# Example patient data
patient_data = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DPF': 0.627,
    'Age': 50
}

# Generate explanation
explanation = explain_prediction(patient_data)

# Access results
print(f"Prediction: {explanation['prediction_label']}")
print(f"Probability: {explanation['probability']:.2%}")
print(f"\n{explanation['explanation_text']}")
print(f"\nTop Features: {explanation['top_features']}")

# Feature contributions
for feature, contrib in explanation['feature_contributions'].items():
    print(f"{feature}: {contrib['shap_value']:.4f}")
```

### Return Value Structure

The `explain_prediction()` function returns a dictionary:

```python
{
    'prediction': 0 or 1,                    # Binary prediction
    'prediction_label': 'Diabetic' or 'Non-Diabetic',
    'probability': 0.0 to 1.0,               # Probability of diabetes
    'explanation_text': 'Human-friendly text explanation',
    'top_features': ['Feature1', 'Feature2', ...],  # Top 5 features
    'feature_contributions': {
        'FeatureName': {
            'shap_value': float,              # SHAP contribution value
            'feature_value': float,          # Actual feature value
            'contribution': float            # Same as shap_value
        },
        ...
    },
    'shap_values': [array]                   # Raw SHAP values (optional)
}
```

### Generating Visualizations

```python
from xai import (
    load_model_and_data, create_shap_explainer,
    plot_shap_summary, plot_shap_force, plot_shap_waterfall,
    plot_global_feature_importance
)
import pandas as pd

# Load model
model, preprocessing_params, feature_names = load_model_and_data()

# Load background data (your training data)
df = pd.read_csv('kaggle_diabetes.csv')
X = df.drop(columns='Outcome')
X_background = X.sample(100, random_state=42)

# Generate visualizations
plot_global_feature_importance(model, feature_names)
plot_shap_summary(model, X_background.values, feature_names)

# For single prediction visualizations
explainer = create_shap_explainer(model)
example_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
plot_shap_force(explainer, example_input, feature_names)
plot_shap_waterfall(explainer, example_input, feature_names)
```

## 🔌 API Reference

### Main Functions

#### `explain_prediction(input_data, ...)`

Generate explanation for a single prediction.

**Parameters:**
- `input_data` (dict or array): Patient data
- `model` (optional): Trained model (loads from file if None)
- `preprocessing_params` (optional): Preprocessing parameters
- `feature_names` (optional): List of feature names
- `background_data` (optional): Background data for SHAP
- `explainer` (optional): Pre-created SHAP explainer
- `return_shap_values` (bool): Whether to return raw SHAP values

**Returns:** Dictionary with prediction and explanation

#### `load_model_and_data()`

Load model, preprocessing parameters, and feature names from files.

**Returns:** `(model, preprocessing_params, feature_names)`

#### `create_shap_explainer(model, background_data=None)`

Create a SHAP explainer for the model.

**Returns:** SHAP explainer object

### Visualization Functions

- `plot_shap_summary(model, X_background, feature_names, save_path=None)`
- `plot_shap_force(explainer, input_data, feature_names, save_path=None)`
- `plot_shap_waterfall(explainer, input_data, feature_names, save_path=None)`
- `plot_global_feature_importance(model, feature_names, save_path=None)`

## 🎨 Visualizations

The system generates several types of visualizations:

1. **SHAP Summary Plot**: Shows global feature importance across all samples
2. **SHAP Force Plot**: Visualizes how each feature pushes the prediction
3. **SHAP Waterfall Plot**: Shows the cumulative effect of features
4. **Global Feature Importance**: Bar chart of feature importances from the model

All visualizations are saved to the `xai_visualizations/` directory.

## 🌐 Web Integration

### Using the XAI-Enabled Flask App

1. **Start the app:**
   ```bash
   python app_xai.py
   ```

2. **Make predictions:**
   - Fill in the form at `http://localhost:5000`
   - Submit to get prediction with explanation

3. **API Endpoint:**
   ```bash
   curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{
       "pregnancies": 6,
       "glucose": 148,
       "bloodpressure": 72,
       "skinthickness": 35,
       "insulin": 0,
       "bmi": 33.6,
       "dpf": 0.627,
       "age": 50
     }'
   ```

### Dataset with Gender Feature

If your dataset includes a Gender column:

1. The model will automatically detect and encode it
2. Add Gender field to your input form in `templates/index.html`
3. The XAI system will include Gender in explanations

## 🔧 Troubleshooting

### Issue: SHAP not installed

**Error:** `ImportError: No module named 'shap'`

**Solution:**
```bash
pip install shap
```

### Issue: Model file not found

**Error:** `FileNotFoundError: diabetes-prediction-xai-model.pkl`

**Solution:**
```bash
python model.py  # Train the model first
```

### Issue: Slow SHAP calculations

**Solutions:**
- Use TreeExplainer (automatic for Random Forest/XGBoost)
- Reduce `SHAP_BACKGROUND_SAMPLES` in `xai.py`
- Use a smaller background dataset

### Issue: Visualization errors

**Error:** `AttributeError: 'Explanation' object has no attribute...`

**Solution:** Update SHAP to latest version:
```bash
pip install --upgrade shap
```

### Issue: Gender encoding errors

**Solution:** Ensure Gender column contains values like:
- 'Male'/'Female'
- 'M'/'F'
- 0/1

The system will automatically encode them.

## 📊 Understanding SHAP Values

- **Positive SHAP value**: Feature increases diabetes risk
- **Negative SHAP value**: Feature decreases diabetes risk
- **Magnitude**: Larger absolute values = stronger effect
- **Sum of SHAP values**: Equals prediction - baseline

## 🎓 Example Explanations

### Example 1: High-Risk Patient

```
Prediction: Diabetic (Confidence: 85.3%)

Key Factors:
1. Your glucose level (148 mg/dL) strongly increased your diabetes risk.
2. Your BMI (33.6 kg/m²) moderately increased your diabetes risk.
3. Your age (50 years) slightly increased your diabetes risk.
```

### Example 2: Low-Risk Patient

```
Prediction: Non-Diabetic (Confidence: 92.1%)

Key Factors:
1. Your glucose level (85 mg/dL) strongly decreased your diabetes risk.
2. Your BMI (26.6 kg/m²) slightly decreased your diabetes risk.
3. Your age (31 years) moderately decreased your diabetes risk.
```

## 🔄 Migration from Legacy Model

If you have an existing model (`diabetes-prediction-rfc-model.pkl`):

1. The system will automatically use it if the new model isn't found
2. For full XAI support, retrain using `model.py`
3. The new model will be saved as `diabetes-prediction-xai-model.pkl`

## 📝 Notes

- SHAP works best with tree-based models (Random Forest, XGBoost)
- For other model types, KernelExplainer is used (slower)
- Visualizations require matplotlib
- All explanations are generated in real-time
- The system handles missing values automatically

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review example_usage.py for code examples
3. Ensure all dependencies are installed correctly

---

**Happy Explaining! 🎉**

