"""
Example: How to use the model evaluation script

This demonstrates three ways to evaluate your trained diabetes prediction model.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluate_model import evaluate_diabetes_model, load_model_and_test_data


# ============================================================================
# EXAMPLE 1: Load model and test data from files
# ============================================================================

def example_1_load_from_files():
    """Load model and test data from saved files."""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Loading model and test data from files")
    print("="*80)
    
    # Load model and prepare test data
    model, X_test, y_test = load_model_and_test_data(
        model_path='diabetes-prediction-xai-model.pkl',
        dataset_path='kaggle_diabetes.csv',
        test_size=0.2,
        random_state=42
    )
    
    # Evaluate the model
    results = evaluate_diabetes_model(model, X_test, y_test)
    
    return results


# ============================================================================
# EXAMPLE 2: Use model and test data already in memory (from training)
# ============================================================================

def example_2_use_existing_data():
    """Use model and test data that are already loaded in memory."""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Using model and test data from training")
    print("="*80)
    
    # Load the trained model
    model = joblib.load('diabetes-prediction-xai-model.pkl')
    
    # Load and prepare test data (same preprocessing as training)
    df = pd.read_csv('kaggle_diabetes.csv')
    if 'DiabetesPedigreeFunction' in df.columns:
        df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
    
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    
    # Split data (same random_state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Evaluate the model
    results = evaluate_diabetes_model(model, X_test, y_test)
    
    return results


# ============================================================================
# EXAMPLE 3: Evaluate after training (integrated with model.py)
# ============================================================================

def example_3_integrated_evaluation():
    """
    Example of how to integrate evaluation into your training script.
    
    This would be added to model.py after training:
    """
    
    code_example = """
    # In model.py, after training:
    
    from evaluate_model import evaluate_diabetes_model
    
    # Train model (existing code)
    model = train_model(X_train, y_train, model_type='random_forest')
    
    # Evaluate model (new code)
    results = evaluate_diabetes_model(model, X_test, y_test)
    
    # Access individual metrics
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    """
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Integration with training script")
    print("="*80)
    print(code_example)


# ============================================================================
# EXAMPLE 4: Access individual metrics from results
# ============================================================================

def example_4_access_metrics():
    """Demonstrate how to access individual metrics from results dictionary."""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Accessing individual metrics")
    print("="*80)
    
    # Load and evaluate
    model, X_test, y_test = load_model_and_test_data()
    results = evaluate_diabetes_model(model, X_test, y_test, display_results=False)
    
    # Access metrics
    print("\nIndividual Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
    
    # Access predictions
    print(f"\nPredictions shape: {results['y_pred'].shape}")
    print(f"Probabilities shape: {results['y_prob'].shape}")
    
    # Access confusion matrix
    cm = results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    """
    Run examples to demonstrate model evaluation usage.
    """
    
    print("\n" + "="*80)
    print("MODEL EVALUATION EXAMPLES")
    print("="*80)
    
    try:
        # Run Example 1: Load from files
        results = example_1_load_from_files()
        
        # Uncomment to run other examples:
        # example_2_use_existing_data()
        # example_3_integrated_evaluation()
        # example_4_access_metrics()
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Model file exists: diabetes-prediction-xai-model.pkl")
        print("2. Dataset file exists: kaggle_diabetes.csv")
        print("3. Run 'python model.py' first to train the model")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

