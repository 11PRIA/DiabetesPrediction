# Quick Start Guide - Explainable AI (XAI) System

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python model.py
```

**Note:** Make sure your dataset file (`kaggle_diabetes.csv` by default) is in the project directory.

### Step 3: Test the System

```bash
python example_usage.py
```

You should see:
- ✓ Model loaded successfully
- ✓ Prediction generated
- ✓ Explanation created

### Step 4: Run the Web App

```bash
python app_xai.py
```

Open `http://localhost:5000` in your browser.

## 📝 Basic Usage

### Python Code Example

```python
from xai import explain_prediction

# Patient data
patient = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DPF': 0.627,
    'Age': 50
}

# Get explanation
result = explain_prediction(patient)

# Print results
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['probability']:.1%}")
print(f"\n{result['explanation_text']}")
```

### Expected Output

```
Prediction: Diabetic
Confidence: 85.3%

The model predicts: **Diabetic** (confidence: 85.3%)

Key Factors:
1. Your glucose level (148 mg/dL) strongly increased your diabetes risk.
2. Your BMI (33.6 kg/m²) moderately increased your diabetes risk.
3. Your age (50 years) slightly increased your diabetes risk.
```

## 🎯 Key Features

✅ **SHAP Integration** - Feature importance and per-patient explanations  
✅ **Human-Friendly Text** - Easy-to-understand explanations  
✅ **Visualizations** - Multiple plot types for analysis  
✅ **Web Integration** - Built into Flask app  
✅ **Gender Support** - Handles datasets with/without Gender feature  

## 📁 Important Files

- `model.py` - Train model with XAI support
- `xai.py` - XAI explanation module
- `app_xai.py` - Web app with XAI
- `example_usage.py` - Usage examples

## ⚠️ Common Issues

**Problem:** `ModuleNotFoundError: No module named 'shap'`  
**Solution:** `pip install shap`

**Problem:** `FileNotFoundError: diabetes-prediction-xai-model.pkl`  
**Solution:** Run `python model.py` first

**Problem:** Model predictions don't match  
**Solution:** Ensure you're using the same preprocessing as training

## 📚 More Information

See `XAI_README.md` for detailed documentation.

---

**Ready to explain your predictions! 🎉**

