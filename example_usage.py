"""
Example Usage Script for Diabetes Prediction with Explainable AI (XAI)

This script demonstrates how to use the XAI system to make predictions
and generate explanations.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import xai
from xai import explain_prediction, load_model_and_data, create_shap_explainer
from xai import plot_shap_summary, plot_shap_force, plot_shap_waterfall, plot_global_feature_importance

# ============================================================================
# EXAMPLE 1: Basic Prediction with Explanation
# ============================================================================

def example_basic_prediction():
    """Example of making a prediction with explanation."""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Prediction with Explanation")
    print("="*70)
    
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
    
    print(f"\nPatient Data:")
    for key, value in patient_data.items():
        print(f"  {key}: {value}")
    
    # Generate explanation
    explanation = explain_prediction(patient_data)
    
    # Display results
    print(f"\n{'='*70}")
    print("PREDICTION RESULT")
    print("="*70)
    print(f"Prediction: {explanation['prediction_label']}")
    print(f"Confidence: {explanation['probability']:.2%}")
    
    print(f"\n{'='*70}")
    print("EXPLANATION")
    print("="*70)
    print(explanation['explanation_text'])
    
    print(f"\n{'='*70}")
    print("TOP CONTRIBUTING FEATURES")
    print("="*70)
    for i, feature in enumerate(explanation['top_features'], 1):
        contrib = explanation['feature_contributions'][feature]
        direction = "↑ increases risk" if contrib['shap_value'] > 0 else "↓ decreases risk"
        print(f"{i}. {feature:20s}: {abs(contrib['shap_value']):.4f} ({direction})")
    
    return explanation

# ============================================================================
# EXAMPLE 2: Multiple Predictions
# ============================================================================

def example_multiple_predictions():
    """Example of making multiple predictions."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Predictions")
    print("="*70)
    
    # Multiple patient examples
    patients = [
        {
            'name': 'Patient A',
            'data': {
                'Pregnancies': 1,
                'Glucose': 85,
                'BloodPressure': 66,
                'SkinThickness': 29,
                'Insulin': 0,
                'BMI': 26.6,
                'DPF': 0.351,
                'Age': 31
            }
        },
        {
            'name': 'Patient B',
            'data': {
                'Pregnancies': 8,
                'Glucose': 183,
                'BloodPressure': 64,
                'SkinThickness': 0,
                'Insulin': 0,
                'BMI': 23.3,
                'DPF': 0.672,
                'Age': 32
            }
        },
        {
            'name': 'Patient C',
            'data': {
                'Pregnancies': 1,
                'Glucose': 89,
                'BloodPressure': 66,
                'SkinThickness': 23,
                'Insulin': 94,
                'BMI': 28.1,
                'DPF': 0.167,
                'Age': 21
            }
        }
    ]
    
    results = []
    
    for patient in patients:
        print(f"\n{'-'*70}")
        print(f"Processing {patient['name']}...")
        print("-"*70)
        
        explanation = explain_prediction(patient['data'])
        results.append({
            'name': patient['name'],
            'prediction': explanation['prediction_label'],
            'probability': explanation['probability'],
            'top_feature': explanation['top_features'][0] if explanation['top_features'] else 'N/A'
        })
        
        print(f"Prediction: {explanation['prediction_label']} ({explanation['probability']:.2%})")
        print(f"Top contributing feature: {results[-1]['top_feature']}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    for result in results:
        print(f"{result['name']:15s}: {result['prediction']:15s} "
              f"({result['probability']:6.2%}) - Top feature: {result['top_feature']}")
    
    return results

# ============================================================================
# EXAMPLE 3: Generate Visualizations
# ============================================================================

def example_generate_visualizations():
    """Example of generating SHAP visualizations."""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Generating Visualizations")
    print("="*70)
    
    # Load model and data
    model, preprocessing_params, feature_names = load_model_and_data()
    
    # Load training data for background (you would load your actual training data)
    # For this example, we'll create a sample
    try:
        import pandas as pd
        df = pd.read_csv('kaggle_diabetes.csv')
        if 'DiabetesPedigreeFunction' in df.columns:
            df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
        
        # Preprocess (simplified - you should use the same preprocessing as training)
        X = df.drop(columns='Outcome')
        if 'Gender' in X.columns:
            # Handle gender encoding if present
            pass
        
        # Use a sample for background
        X_background = X.sample(min(100, len(X)), random_state=42)
        
        print("\nGenerating visualizations...")
        print("This may take a few moments...")
        
        # Generate global feature importance
        plot_global_feature_importance(model, feature_names)
        
        # Generate SHAP summary plot
        plot_shap_summary(model, X_background.values, feature_names)
        
        # Example single prediction for force and waterfall plots
        example_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
        explainer = create_shap_explainer(model)
        
        plot_shap_force(explainer, example_input, feature_names)
        plot_shap_waterfall(explainer, example_input, feature_names)
        
        print("\n✓ All visualizations generated successfully!")
        print(f"✓ Check the 'xai_visualizations' directory for saved plots")
        
    except FileNotFoundError:
        print("⚠ Dataset file not found. Skipping visualization generation.")
        print("  To generate visualizations, ensure 'kaggle_diabetes.csv' is available.")
    except Exception as e:
        print(f"⚠ Error generating visualizations: {e}")

# ============================================================================
# EXAMPLE 4: Detailed Feature Analysis
# ============================================================================

def example_detailed_analysis():
    """Example of detailed feature contribution analysis."""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Detailed Feature Analysis")
    print("="*70)
    
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
    
    explanation = explain_prediction(patient_data)
    
    print(f"\nPrediction: {explanation['prediction_label']} "
          f"(Probability: {explanation['probability']:.4f})")
    
    print(f"\n{'='*70}")
    print("DETAILED FEATURE CONTRIBUTIONS")
    print("="*70)
    print(f"{'Feature':<20s} {'Value':<12s} {'SHAP Value':<15s} {'Contribution'}")
    print("-"*70)
    
    # Sort by absolute SHAP value
    sorted_features = sorted(
        explanation['feature_contributions'].items(),
        key=lambda x: abs(x[1]['shap_value']),
        reverse=True
    )
    
    for feature, contrib in sorted_features:
        shap_val = contrib['shap_value']
        feature_val = contrib['feature_value']
        
        if shap_val > 0:
            contribution = f"+{shap_val:.4f} (increases risk)"
        else:
            contribution = f"{shap_val:.4f} (decreases risk)"
        
        print(f"{feature:<20s} {feature_val:<12.2f} {shap_val:<15.4f} {contribution}")
    
    # Risk factors summary
    print(f"\n{'='*70}")
    print("RISK FACTORS SUMMARY")
    print("="*70)
    
    risk_increasing = [
        (feat, contrib['shap_value'])
        for feat, contrib in explanation['feature_contributions'].items()
        if contrib['shap_value'] > 0
    ]
    risk_decreasing = [
        (feat, contrib['shap_value'])
        for feat, contrib in explanation['feature_contributions'].items()
        if contrib['shap_value'] < 0
    ]
    
    if risk_increasing:
        print("\nFactors INCREASING diabetes risk:")
        for feat, val in sorted(risk_increasing, key=lambda x: x[1], reverse=True):
            print(f"  • {feat}: +{val:.4f}")
    
    if risk_decreasing:
        print("\nFactors DECREASING diabetes risk:")
        for feat, val in sorted(risk_decreasing, key=lambda x: x[1]):
            print(f"  • {feat}: {val:.4f}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DIABETES PREDICTION XAI - EXAMPLE USAGE")
    print("="*70)
    print("\nThis script demonstrates various ways to use the XAI system.")
    print("Uncomment the examples you want to run.\n")
    
    # Run examples
    try:
        # Example 1: Basic prediction
        example_basic_prediction()
        
        # Example 2: Multiple predictions
        # example_multiple_predictions()
        
        # Example 3: Generate visualizations
        # example_generate_visualizations()
        
        # Example 4: Detailed analysis
        # example_detailed_analysis()
        
    except FileNotFoundError as e:
        print(f"\n⚠ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Trained the model: python model.py")
        print("  2. The model files are in the current directory")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()

