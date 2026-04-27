# Recommendation Engine Documentation

## Overview

The Recommendation Engine generates personalized dietary and exercise plans based on:
- **Diabetes risk prediction** (probability and risk level)
- **SHAP-based feature contributions** (explaining WHY the risk exists)
- **User preferences** (diet type, activity level, time availability)

**⚠️ IMPORTANT DISCLAIMER:**
This is a simplified educational helper only. It is **NOT medical advice**. It does **NOT** handle:
- Allergies
- Specific medical histories
- Medication interactions
- Complex health conditions

Always consult with healthcare professionals for personalized medical advice.

## Quick Start

### 1. Test the Recommendation Engine

```bash
python test_recommendation.py
```

### 2. Use in Your Code

```python
from xai import explain_prediction, load_model_and_data
from recommendation import generate_recommendation, generate_human_friendly_summary

# Load model
model, preprocessing_params, feature_names = load_model_and_data()

# Patient data
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

# User preferences
preferences = {
    'diet_pref': 'vegetarian',      # 'vegetarian'|'non-veg'|'vegan'
    'activity_level': 'sedentary',   # 'sedentary'|'light'|'moderate'|'active'
    'time_per_day_min': 30          # minutes available per day
}

# Generate XAI explanation
xai_summary = explain_prediction(
    patient_data,
    model=model,
    preprocessing_params=preprocessing_params,
    feature_names=feature_names
)

# Generate recommendations
recommendation = generate_recommendation(
    patient_data,
    xai_summary,
    preferences
)

# Get human-friendly summary
summary = generate_human_friendly_summary(recommendation)
print(summary)
```

## API Reference

### `generate_recommendation(input_dict, xai_summary, preferences)`

Generate personalized recommendations.

**Parameters:**
- `input_dict` (dict): Patient features (glucose, BMI, age, etc.)
- `xai_summary` (dict): Output from `explain_prediction()` containing:
  - `probability`: float (0-1)
  - `top_features`: list of feature names
  - `feature_contributions`: dict with SHAP values
- `preferences` (dict): User preferences:
  - `diet_pref`: 'vegetarian'|'non-veg'|'vegan'
  - `activity_level`: 'sedentary'|'light'|'moderate'|'active'
  - `time_per_day_min`: int (minutes available)

**Returns:**
Dictionary with:
- `risk_bucket`: "Low"|"Moderate"|"High"
- `probability`: float
- `priority_interventions`: list of intervention priorities
- `diet_plan`: dict with daily guidelines and weekly focus
- `exercise_plan`: dict with weekly schedule and goals
- `nudges`: list of behavioral tips
- `disclaimer`: medical disclaimer text

### `generate_human_friendly_summary(recommendation)`

Generate natural language summary of recommendations.

**Parameters:**
- `recommendation`: Output from `generate_recommendation()`

**Returns:**
Human-friendly text string

## Risk Level Buckets

- **Low Risk**: probability < 0.2
- **Moderate Risk**: 0.2 ≤ probability < 0.5
- **High Risk**: probability ≥ 0.5

## Feature-to-Intervention Mapping

The engine maps top contributing features to interventions:

- **Glucose/Blood Sugar** → Carbohydrate moderation, low-GI foods, glucose monitoring
- **BMI/Weight** → Weight loss, calorie deficit, cardio + strength training
- **Blood Pressure** → Salt reduction, aerobic exercise, stress management
- **Physical Activity** → Structured exercise program, increase daily activity
- **Age** → Age-appropriate exercise, lifestyle modifications

## Diet Plans

### Vegetarian
- Emphasizes legumes, whole grains, low-GI carbs
- Includes meal templates for breakfast/lunch/dinner/snacks
- Focus on portion control and nutrient balance

### Non-Vegetarian
- Same as vegetarian + lean proteins (fish/chicken)
- Avoids processed meats
- Includes fish 2-3 times per week

### Vegan
- Whole-food plant-based templates
- Emphasizes protein sources (tofu/legumes)
- Includes B12 considerations

## Exercise Plans

### Sedentary
- Starts with 10-15 min walking daily
- Progresses 5 min per week to 30 min
- Low-impact, gradual progression

### Light
- 3-4 sessions per week
- 20-30 min moderate activity
- Includes stretching

### Moderate
- 4-5 sessions per week
- 150 min/week moderate aerobic
- 2x strength training sessions

### Active
- 5-6 sessions per week
- 225+ min/week
- Mix of cardio and strength

## Web Integration

The recommendation engine is integrated into `app_xai.py`:

1. **Form Fields Added** (in `templates/index.html`):
   - Diet preference dropdown
   - Activity level dropdown
   - Time available input

2. **Result Display** (in `templates/result_xai.html`):
   - Shows priority interventions
   - Displays diet plan with daily guidelines
   - Shows exercise plan with weekly schedule
   - Lists behavioral nudges
   - Includes disclaimer

## Example Output

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
    "daily_guidelines": {
      "breakfast": "Oatmeal with berries and nuts...",
      "lunch": "Lentil soup with whole grain bread...",
      "dinner": "Grilled vegetables with tofu...",
      "snacks": "Apple with almond butter..."
    },
    "weekly_focus": [
      "Reduce refined carbohydrates and added sugars",
      "Create moderate calorie deficit"
    ]
  },
  "exercise_plan": {
    "summary": "Start with gentle walking...",
    "weekly_plan": [
      {"day": "Monday", "activity": "10-minute slow walk", "type": "cardio"},
      {"day": "Wednesday", "activity": "10-minute slow walk", "type": "cardio"},
      {"day": "Friday", "activity": "10-minute slow walk", "type": "cardio"}
    ],
    "goals": {
      "weekly_minutes": 30,
      "strength_sessions_per_week": 0,
      "daily_steps_target": 8000
    }
  },
  "nudges": [
    "Weigh yourself once per week",
    "Track daily steps",
    "Read food labels for carbohydrate content"
  ]
}
```

## Testing

Run the test script:

```bash
python test_recommendation.py
```

This will:
1. Load the model
2. Generate XAI explanation
3. Create recommendations
4. Display JSON and human-friendly output

## Limitations

1. **Educational Only**: Not a substitute for medical advice
2. **No Medical History**: Doesn't account for existing conditions
3. **No Allergies**: Doesn't handle food allergies
4. **No Medications**: Doesn't consider medication interactions
5. **Simplified Rules**: Uses rule-based logic, not personalized medical protocols

## Future Enhancements

- Integration with nutrition databases
- Meal planning with recipes
- Progress tracking
- Integration with fitness apps
- More granular dietary restrictions

---

**Remember: Always consult healthcare professionals for personalized medical advice.**

