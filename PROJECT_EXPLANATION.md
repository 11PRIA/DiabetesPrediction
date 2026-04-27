# Complete Project Explanation & Running Guide

## 📋 Project Overview

This is a **Diabetes Prediction System with Explainable AI (XAI)** and **Personalized Recommendations**. It's a full-stack Flask web application that:

1. **Predicts diabetes risk** from clinical parameters (Glucose, BMI, Age, etc.)
2. **Explains predictions** using SHAP (SHapley Additive exPlanations) to show which features contributed most
3. **Generates personalized recommendations** for diet, exercise, and lifestyle based on:
   - Risk level (Low/Moderate/High)
   - Top contributing factors
   - User preferences (diet type, activity level, time available)

---

## 🏗️ Project Architecture

### Core Components

```
Diabetes-Prediction/
├── app_xai.py              # Main Flask web app (with XAI + recommendations)
├── app.py                  # Legacy Flask app (basic prediction only)
├── model.py                # Model training script (Random Forest/XGBoost)
├── xai.py                  # SHAP-based explainability module
├── recommendation.py        # Personalized diet/exercise recommendation engine
├── kaggle_diabetes.csv     # Training dataset (769 samples)
├── requirements.txt        # Python dependencies
└── templates/              # HTML templates for web UI
    ├── index.html         # Input form
    ├── result.html        # Basic results page
    └── result_xai.html   # Results with XAI explanations
```

### Data Flow

```
User Input → Preprocessing → Model Prediction → SHAP Explanation → Recommendations → Web UI/API
```

---

## 📊 Dataset Structure

The dataset (`kaggle_diabetes.csv`) contains:
- **Pregnancies**: Number of pregnancies
- **Glucose**: Blood glucose level (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body Mass Index (kg/m²)
- **DiabetesPedigreeFunction (DPF)**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Target variable (0 = Non-Diabetic, 1 = Diabetic)

**Note:** Missing values are represented as 0 in the dataset and are imputed during preprocessing.

---

## 🔧 How to Run the Project

### Prerequisites

- Python 3.7+ installed
- Virtual environment (recommended)

### Step 1: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- Flask (web framework)
- scikit-learn (machine learning)
- pandas, numpy (data processing)
- SHAP (explainable AI)
- joblib (model serialization)
- matplotlib (visualizations)

### Step 3: Train the Model

**IMPORTANT:** You must train the model before running the web app!

```bash
python model.py
```

**What this does:**
- Loads `kaggle_diabetes.csv`
- Preprocesses data (handles missing values, encodes features)
- Trains a Random Forest or XGBoost classifier
- Saves:
  - `diabetes-prediction-xai-model.pkl` (trained model)
  - `preprocessing_params.pkl` (mean/median values for imputation)
  - `feature_names.pkl` (feature order)

**Expected output:**
```
✓ Dataset loaded successfully: 769 rows
✓ Model training completed!
✓ Model saved to: diabetes-prediction-xai-model.pkl
```

### Step 4: Run the Web Application

**Option A: Full-featured app (with XAI + Recommendations)**
```bash
python app_xai.py
```

**Option B: Basic app (prediction only)**
```bash
python app.py
```

**Access the app:**
- Open your browser and go to: `http://localhost:5000`
- You'll see a form to input patient data
- Submit to get predictions with explanations

### Step 5: Test the API (Optional)

The app also provides a JSON API endpoint:

```bash
# Using curl (Windows PowerShell)
curl -X POST http://localhost:5000/api/predict `
  -H "Content-Type: application/json" `
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "bloodpressure": 72,
    "skinthickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "dpf": 0.627,
    "age": 50,
    "diet_pref": "non-veg",
    "activity_level": "moderate",
    "time_per_day_min": 30
  }'
```

---

## 🧠 How the Project Works

### 1. Data Preprocessing (`model.py`)

**Handles missing values:**
- Replaces 0 values with NaN for: Glucose, BloodPressure, SkinThickness, Insulin, BMI
- Imputes missing values:
  - Glucose, BloodPressure → **Mean**
  - SkinThickness, Insulin, BMI → **Median**
- Stores imputation values in `preprocessing_params.pkl` for consistent preprocessing during prediction

**Feature encoding:**
- Gender (if present): Male=1, Female=0
- All features normalized to numeric format

### 2. Model Training (`model.py`)

**Algorithm:** Random Forest Classifier (default) or XGBoost (optional)

**Configuration:**
- Train/Test split: 80/20 (stratified)
- Random state: 42 (for reproducibility)
- Hyperparameters: Configurable in `model.py`

**Evaluation metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix

### 3. Prediction (`app_xai.py`)

**Process:**
1. User submits form with clinical parameters
2. Input is preprocessed (same as training):
   - Replace 0s with stored mean/median values
   - Ensure correct feature order
3. Model predicts probability of diabetes (0-1)
4. Prediction threshold: 0.5 (default)

### 4. Explainable AI (`xai.py`)

**SHAP Integration:**
- Uses **TreeExplainer** (fast for tree-based models)
- Calculates SHAP values for each feature
- Shows how each feature contributes to the prediction

**Output includes:**
- **Top 5 features** contributing most to the prediction
- **Feature contributions** (positive = increases risk, negative = decreases risk)
- **Human-friendly explanation text**
- Optional visualizations (summary plots, force plots, waterfall plots)

**Example explanation:**
```
The model predicts: **Diabetic** (confidence: 85.3%)

Key Factors:
1. Your glucose level (148 mg/dL) strongly increased your diabetes risk.
2. Your BMI (33.6 kg/m²) moderately increased your diabetes risk.
3. Your age (50 years) slightly increased your diabetes risk.
```

### 5. Recommendation Engine (`recommendation.py`)

**Risk Buckets:**
- **Low:** Probability < 0.2
- **Moderate:** 0.2 ≤ Probability < 0.5
- **High:** Probability ≥ 0.5

**Personalization based on:**
- **Risk level** → Intensity of recommendations
- **Top SHAP features** → Priority interventions
  - High Glucose → Carbohydrate moderation, low-GI foods
  - High BMI → Weight loss, calorie deficit, cardio + strength training
  - High Blood Pressure → Salt reduction, aerobic exercise
- **User preferences:**
  - Diet type: vegetarian, non-veg, vegan
  - Activity level: sedentary, light, moderate, active
  - Time available: minutes per day

**Output includes:**
- **Diet plan:** Daily meal suggestions, weekly focus, general guidelines
- **Exercise plan:** Weekly schedule, goals (minutes/week, steps/day, strength sessions)
- **Behavioral nudges:** Tracking suggestions, goal-setting tips

**Example recommendation:**
```json
{
  "risk_bucket": "High",
  "probability": 0.93,
  "priority_interventions": [
    "Reduce refined carbohydrates",
    "Weight loss and calorie management",
    "Increase cardiovascular exercise"
  ],
  "diet_plan": {
    "summary": "Based on your high diabetes risk...",
    "daily_guidelines": {...},
    "weekly_focus": [...]
  },
  "exercise_plan": {
    "summary": "Start with gentle walking...",
    "weekly_plan": [...],
    "goals": {
      "weekly_minutes": 180,
      "daily_steps_target": 8000
    }
  }
}
```

---

## 📁 Key Files Explained

### `app_xai.py` (Main Application)
- Flask web server
- Routes:
  - `/` → Home page (input form)
  - `/predict` → Handle form submission (POST)
  - `/api/predict` → JSON API endpoint
- Integrates XAI and recommendations
- Falls back to basic prediction if XAI unavailable

### `model.py` (Training Script)
- Loads and preprocesses dataset
- Trains Random Forest or XGBoost
- Evaluates model performance
- Saves model and preprocessing parameters

### `xai.py` (Explainability Module)
- `explain_prediction()` → Main function for generating explanations
- `load_model_and_data()` → Loads saved model
- `preprocess_input()` → Preprocesses user input
- Visualization functions (summary, force, waterfall plots)

### `recommendation.py` (Recommendation Engine)
- `generate_recommendation()` → Main function
- `generate_human_friendly_summary()` → Text summary
- Diet templates (vegetarian, non-veg, vegan)
- Exercise templates (sedentary, light, moderate, active)

---

## 🎯 Usage Examples

### Example 1: Basic Prediction (Python)

```python
from xai import explain_prediction

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

result = explain_prediction(patient)
print(f"Prediction: {result['prediction_label']}")
print(f"Probability: {result['probability']:.2%}")
print(f"\n{result['explanation_text']}")
```

### Example 2: With Recommendations

```python
from xai import explain_prediction
from recommendation import generate_recommendation

# Get explanation
explanation = explain_prediction(patient)

# Generate recommendations
preferences = {
    'diet_pref': 'vegetarian',
    'activity_level': 'moderate',
    'time_per_day_min': 30
}

recommendation = generate_recommendation(
    patient,
    explanation,
    preferences
)

print(recommendation['diet_plan']['summary'])
print(recommendation['exercise_plan']['summary'])
```

---

## ⚠️ Important Notes

### Medical Disclaimer
**This is an educational tool, NOT medical advice.**
- Does not handle allergies, medications, or medical history
- Always consult healthcare professionals for real medical decisions
- Model accuracy depends on training data quality

### Model Requirements
- Must train model (`python model.py`) before running web app
- Model file must exist: `diabetes-prediction-xai-model.pkl`
- Preprocessing parameters must exist: `preprocessing_params.pkl`

### Troubleshooting

**Error: Model file not found**
```bash
# Solution: Train the model first
python model.py
```

**Error: SHAP not installed**
```bash
# Solution: Install SHAP
pip install shap
```

**Error: Port already in use**
```python
# Edit app_xai.py, change port:
app.run(debug=True, port=5001)  # Use different port
```

**Model predictions seem wrong**
- Ensure preprocessing matches training (check `preprocessing_params.pkl`)
- Verify feature order matches `feature_names.pkl`

---

## 🔄 Workflow Summary

1. **Setup:** Install dependencies, activate virtual environment
2. **Train:** Run `python model.py` to train and save model
3. **Run:** Execute `python app_xai.py` to start web server
4. **Use:** Open browser → Enter patient data → Get prediction + explanation + recommendations
5. **API:** Use `/api/predict` endpoint for programmatic access

---

## 📈 Model Performance

The model typically achieves:
- **Accuracy:** ~75-80%
- **Precision:** ~70-75%
- **Recall:** ~60-70%
- **F1-Score:** ~65-70%

*Note: Performance varies based on dataset and hyperparameters*

---

## 🚀 Next Steps

1. **Improve model:** Tune hyperparameters, try different algorithms
2. **Add features:** Include more clinical parameters
3. **Enhance UI:** Improve web interface design
4. **Deploy:** Deploy to cloud (Heroku, AWS, etc.)
5. **Expand recommendations:** Add more personalized suggestions

---

## 📚 Additional Resources

- `README.md` - Project overview
- `QUICK_START_XAI.md` - Quick setup guide
- `XAI_README.md` - Detailed XAI documentation
- `HOW_TO_IMPROVE_METRICS.md` - Model improvement tips

---

**Ready to predict and explain diabetes risk! 🎉**

