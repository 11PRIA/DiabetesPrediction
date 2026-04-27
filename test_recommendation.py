"""
Test script for the Recommendation Engine

This demonstrates how to use the recommendation engine with the XAI system.
"""

from xai import explain_prediction, load_model_and_data
from recommendation import generate_recommendation, generate_human_friendly_summary
import json

def test_recommendation_system():
    """Test the recommendation system end-to-end."""
    
    print("\n" + "="*70)
    print("TESTING RECOMMENDATION ENGINE")
    print("="*70)
    
    # Load model
    print("\n1. Loading model...")
    model, preprocessing_params, feature_names = load_model_and_data()
    
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
    
    print("\n2. Patient Data:")
    for key, value in patient_data.items():
        print(f"   {key}: {value}")
    
    # User preferences
    preferences = {
        'diet_pref': 'vegetarian',
        'activity_level': 'sedentary',
        'time_per_day_min': 30
    }
    
    print("\n3. User Preferences:")
    for key, value in preferences.items():
        print(f"   {key}: {value}")
    
    # Generate XAI explanation
    print("\n4. Generating XAI explanation...")
    xai_summary = explain_prediction(
        patient_data,
        model=model,
        preprocessing_params=preprocessing_params,
        feature_names=feature_names
    )
    
    print(f"   Prediction: {xai_summary['prediction_label']}")
    print(f"   Probability: {xai_summary['probability']:.2%}")
    print(f"   Top Features: {xai_summary['top_features']}")
    
    # Generate recommendations
    print("\n5. Generating recommendations...")
    recommendation = generate_recommendation(
        patient_data,
        xai_summary,
        preferences
    )
    
    # Display results
    print("\n" + "="*70)
    print("RECOMMENDATION RESULTS")
    print("="*70)
    
    print(f"\nRisk Level: {recommendation['risk_bucket']}")
    print(f"Probability: {recommendation['probability']:.2%}")
    print(f"\nPriority Interventions:")
    for i, interv in enumerate(recommendation['priority_interventions'], 1):
        print(f"   {i}. {interv}")
    
    print(f"\nDiet Plan ({recommendation['diet_plan']['diet_type']}):")
    print(f"   Summary: {recommendation['diet_plan']['summary']}")
    print(f"\n   Daily Guidelines:")
    for meal, guideline in recommendation['diet_plan']['daily_guidelines'].items():
        print(f"      {meal.capitalize()}: {guideline}")
    
    print(f"\n   Weekly Focus:")
    for focus in recommendation['diet_plan']['weekly_focus']:
        print(f"      • {focus}")
    
    print(f"\nExercise Plan:")
    print(f"   Summary: {recommendation['exercise_plan']['summary']}")
    print(f"\n   Weekly Plan:")
    for day_plan in recommendation['exercise_plan']['weekly_plan']:
        print(f"      {day_plan['day']}: {day_plan['activity']}")
    
    goals = recommendation['exercise_plan']['goals']
    print(f"\n   Goals:")
    print(f"      Weekly Minutes: {goals['weekly_minutes']}")
    print(f"      Strength Sessions: {goals['strength_sessions_per_week']}")
    print(f"      Daily Steps Target: {goals['daily_steps_target']:,}")
    
    print(f"\nTracking & Nudges:")
    for nudge in recommendation['nudges']:
        print(f"   • {nudge}")
    
    print(f"\n⚠️  Disclaimer:")
    print(f"   {recommendation['disclaimer']}")
    
    # Human-friendly summary
    print("\n" + "="*70)
    print("HUMAN-FRIENDLY SUMMARY")
    print("="*70)
    print(generate_human_friendly_summary(recommendation))
    
    # JSON output
    print("\n" + "="*70)
    print("JSON OUTPUT (first 500 chars)")
    print("="*70)
    json_str = json.dumps(recommendation, indent=2)
    print(json_str[:500] + "...")
    
    return recommendation

if __name__ == '__main__':
    test_recommendation_system()

