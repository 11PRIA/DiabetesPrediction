"""
Model Evaluation Script for Diabetes Prediction

This script evaluates a trained Random Forest classifier for binary classification
(Type 2 diabetes prediction: 0 = non-diabetic, 1 = diabetic).

The script computes comprehensive evaluation metrics and displays them in a
clean tabular format suitable for research papers.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')


def evaluate_diabetes_model(model, X_test, y_test, display_results=True):
    """
    Comprehensive evaluation of a trained diabetes prediction model.
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestClassifier
        Trained Random Forest classifier model
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix
    y_test : array-like of shape (n_samples,)
        True binary labels (0 = non-diabetic, 1 = diabetic)
    display_results : bool, default=True
        Whether to print formatted results to console
        
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics and predictions
        - 'y_pred': Predicted class labels
        - 'y_prob': Predicted probabilities for positive class (diabetic)
        - 'accuracy': Accuracy score
        - 'precision': Precision score
        - 'recall': Recall (Sensitivity) score
        - 'f1_score': F1-score
        - 'auc_roc': AUC-ROC score
        - 'confusion_matrix': Confusion matrix array
        - 'classification_report': Text classification report
    """
    
    # ========================================================================
    # STEP 1: Generate Predictions
    # ========================================================================
    
    # Generate binary class predictions (0 or 1)
    y_pred = model.predict(X_test)
    
    # Generate prediction probabilities for the positive class (diabetic)
    # predict_proba returns probabilities for both classes [P(0), P(1)]
    y_prob = model.predict_proba(X_test)[:, 1]  # Extract probability of class 1 (diabetic)
    
    # ========================================================================
    # STEP 2: Compute Evaluation Metrics
    # ========================================================================
    
    # Accuracy: Overall correctness of predictions
    accuracy = accuracy_score(y_test, y_pred)
    
    # Precision: Proportion of positive predictions that are correct
    # Also known as Positive Predictive Value (PPV)
    precision = precision_score(y_test, y_pred, zero_division=0)
    
    # Recall (Sensitivity): Proportion of actual positives correctly identified
    # Also known as True Positive Rate (TPR)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    # F1-Score: Harmonic mean of precision and recall
    # Balances precision and recall, useful for imbalanced datasets
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # AUC-ROC: Area Under the Receiver Operating Characteristic curve
    # Measures the model's ability to distinguish between classes
    # Range: 0.5 (random) to 1.0 (perfect)
    try:
        auc_roc = roc_auc_score(y_test, y_prob)
    except ValueError as e:
        # Handle edge case where only one class is present in y_test
        print(f"Warning: Could not compute AUC-ROC: {e}")
        auc_roc = None
    
    # Confusion Matrix: 2x2 matrix showing TP, FP, TN, FN
    # Format: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification Report: Detailed per-class metrics
    class_report = classification_report(y_test, y_pred, 
                                         target_names=['Non-Diabetic', 'Diabetic'],
                                         output_dict=True)
    
    # ========================================================================
    # STEP 3: Compile Results
    # ========================================================================
    
    results = {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'classification_report': class_report
    }
    
    # ========================================================================
    # STEP 4: Display Results (if requested)
    # ========================================================================
    
    if display_results:
        display_evaluation_results(results, cm)
    
    return results


def display_evaluation_results(results, cm):
    """
    Display evaluation metrics in a clean tabular format suitable for research papers.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing evaluation metrics
    cm : array-like
        Confusion matrix (2x2)
    """
    
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS - DIABETES PREDICTION")
    print("="*80)
    
    # Extract metrics
    accuracy = results['accuracy']
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1_score']
    auc_roc = results['auc_roc']
    
    # ========================================================================
    # Table 1: Performance Metrics
    # ========================================================================
    
    print("\n" + "-"*80)
    print("PERFORMANCE METRICS")
    print("-"*80)
    
    # Create a pandas DataFrame for clean tabular display
    metrics_df = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'Precision',
            'Recall (Sensitivity)',
            'F1-Score',
            'AUC-ROC'
        ],
        'Value': [
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{auc_roc:.4f}" if auc_roc is not None else "N/A"
        ],
        'Interpretation': [
            f"{accuracy*100:.2f}% of predictions are correct",
            f"{precision*100:.2f}% of positive predictions are true positives",
            f"{recall*100:.2f}% of actual positives are correctly identified",
            f"Harmonic mean of precision and recall",
            f"Model distinguishes classes with {auc_roc*100:.2f}% accuracy" if auc_roc else "N/A"
        ]
    })
    
    # Display the table
    print(metrics_df.to_string(index=False))
    
    # ========================================================================
    # Table 2: Confusion Matrix
    # ========================================================================
    
    print("\n" + "-"*80)
    print("CONFUSION MATRIX")
    print("-"*80)
    
    # Extract values from confusion matrix
    # cm format: [[TN, FP], [FN, TP]]
    tn, fp = cm[0, 0], cm[0, 1]  # True Negatives, False Positives
    fn, tp = cm[1, 0], cm[1, 1]  # False Negatives, True Positives
    
    # Create confusion matrix DataFrame
    cm_df = pd.DataFrame(
        cm,
        index=['Actual: Non-Diabetic', 'Actual: Diabetic'],
        columns=['Predicted: Non-Diabetic', 'Predicted: Diabetic']
    )
    
    print(cm_df.to_string())
    
    # Additional statistics from confusion matrix
    print("\n" + "-"*80)
    print("CONFUSION MATRIX STATISTICS")
    print("-"*80)
    
    total = tn + fp + fn + tp
    
    stats_df = pd.DataFrame({
        'Metric': [
            'True Negatives (TN)',
            'False Positives (FP)',
            'False Negatives (FN)',
            'True Positives (TP)',
            'Total Samples'
        ],
        'Count': [tn, fp, fn, tp, total],
        'Percentage': [
            f"{tn/total*100:.2f}%",
            f"{fp/total*100:.2f}%",
            f"{fn/total*100:.2f}%",
            f"{tp/total*100:.2f}%",
            "100.00%"
        ]
    })
    
    print(stats_df.to_string(index=False))
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    print(f"Test Set Size: {total} samples")
    print(f"Correct Predictions: {tn + tp} ({accuracy*100:.2f}%)")
    print(f"Incorrect Predictions: {fp + fn} ({(fp + fn)/total*100:.2f}%)")
    
    if auc_roc is not None:
        # Interpret AUC-ROC
        if auc_roc >= 0.9:
            auc_interpretation = "Excellent"
        elif auc_roc >= 0.8:
            auc_interpretation = "Good"
        elif auc_roc >= 0.7:
            auc_interpretation = "Fair"
        elif auc_roc >= 0.6:
            auc_interpretation = "Poor"
        else:
            auc_interpretation = "Very Poor"
        
        print(f"AUC-ROC Interpretation: {auc_interpretation} ({auc_roc:.4f})")
    
    print("\n" + "="*80 + "\n")


def load_model_and_test_data(model_path='diabetes-prediction-xai-model.pkl',
                             dataset_path='kaggle_diabetes.csv',
                             test_size=0.2,
                             random_state=42):
    """
    Load trained model and prepare test dataset for evaluation.
    
    Parameters:
    -----------
    model_path : str, default='diabetes-prediction-xai-model.pkl'
        Path to the saved model file
    dataset_path : str, default='kaggle_diabetes.csv'
        Path to the dataset CSV file
    test_size : float, default=0.2
        Proportion of dataset to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (model, X_test, y_test)
        Trained model and test features/labels
    """
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import joblib
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print("✓ Model loaded successfully")
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Handle column renaming
    if 'DiabetesPedigreeFunction' in df.columns:
        df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
    
    # Separate features and target
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    
    # Split into train and test sets (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Test set prepared: {len(X_test)} samples")
    
    return model, X_test, y_test


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    """
    Example usage: Evaluate a trained model using saved model file and dataset.
    """
    
    print("\n" + "="*80)
    print("DIABETES PREDICTION MODEL EVALUATION")
    print("="*80)
    
    try:
        # Option 1: Load model and test data automatically
        # Uncomment the following lines if you want to load from files:
        
        # model, X_test, y_test = load_model_and_test_data(
        #     model_path='diabetes-prediction-xai-model.pkl',
        #     dataset_path='kaggle_diabetes.csv'
        # )
        
        # Option 2: Use model and test data already in memory
        # If you have the model and test data from training, use them directly:
        
        # Example: If running after model.py training
        # results = evaluate_diabetes_model(model, X_test, y_test)
        
        print("\n" + "="*80)
        print("USAGE INSTRUCTIONS")
        print("="*80)
        print("\nTo evaluate your model, use one of the following approaches:\n")
        print("APPROACH 1: Load model and test data from files")
        print("-" * 80)
        print("model, X_test, y_test = load_model_and_test_data()")
        print("results = evaluate_diabetes_model(model, X_test, y_test)")
        print("\nAPPROACH 2: Use model and test data already in memory")
        print("-" * 80)
        print("results = evaluate_diabetes_model(model, X_test, y_test)")
        print("\nAPPROACH 3: Import and use in your own script")
        print("-" * 80)
        print("from evaluate_model import evaluate_diabetes_model")
        print("results = evaluate_diabetes_model(model, X_test, y_test)")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Model file exists (diabetes-prediction-xai-model.pkl)")
        print("2. Dataset file exists (kaggle_diabetes.csv)")
        print("3. All dependencies are installed")

