"""
Enhanced Diabetes Prediction Model with Explainable AI (XAI) Support

This script trains a diabetes prediction model (Random Forest or XGBoost) 
and integrates SHAP for explainability. It supports datasets with or without 
a Gender feature.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - EDIT THESE AS NEEDED
# ============================================================================

# Dataset configuration
DATASET_PATH = 'kaggle_diabetes.csv'  # <-- INSERT YOUR DATASET PATH HERE

# Model configuration
MODEL_TYPE = 'random_forest'  # Options: 'random_forest' or 'xgboost'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Random Forest parameters (if MODEL_TYPE == 'random_forest')
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2

# XGBoost parameters (if MODEL_TYPE == 'xgboost')
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

# Output file names
MODEL_FILENAME = 'diabetes-prediction-xai-model.pkl'
PREPROCESSING_FILENAME = 'preprocessing_params.pkl'
FEATURE_NAMES_FILENAME = 'feature_names.pkl'

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess the diabetes dataset.
    
    Args:
        dataset_path: Path to the CSV dataset file
        
    Returns:
        X: Preprocessed feature matrix
        y: Target variable
        feature_names: List of feature names
        preprocessing_params: Dictionary of preprocessing parameters
    """
    print("="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"✓ Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"ERROR: Dataset file '{dataset_path}' not found!")
        print("Please update DATASET_PATH in this script with your dataset location.")
        raise
    
    # Display dataset info
    print(f"\nDataset columns: {list(df.columns)}")
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Rename DiabetesPedigreeFunction to DPF for consistency
    if 'DiabetesPedigreeFunction' in df.columns:
        df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
        print("\n✓ Renamed 'DiabetesPedigreeFunction' to 'DPF'")
    
    # Check for Gender column
    has_gender = 'Gender' in df.columns
    if has_gender:
        print("\n✓ Gender column detected - will be encoded as categorical feature")
    else:
        print("\n⚠ Gender column not found - model will work without it")
    
    # Create a copy for preprocessing
    df_copy = df.copy(deep=True)
    
    # Handle categorical features (Gender)
    if has_gender:
        # Encode Gender: Male=1, Female=0 (or handle other encodings)
        if df_copy['Gender'].dtype == 'object':
            gender_mapping = {'Male': 1, 'Female': 0, 'M': 1, 'F': 0, 'male': 1, 'female': 0}
            df_copy['Gender'] = df_copy['Gender'].map(gender_mapping)
            # Fill any unmapped values with mode
            if df_copy['Gender'].isnull().any():
                mode_gender = df_copy['Gender'].mode()[0]
                df_copy['Gender'].fillna(mode_gender, inplace=True)
                print(f"✓ Encoded Gender column (unmapped values filled with mode: {mode_gender})")
        print(f"✓ Gender encoding complete. Value counts:\n{df_copy['Gender'].value_counts()}")
    
    # Handle missing values (0s in specific columns)
    # Replace 0 values with NaN for columns that shouldn't have 0
    zero_replace_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    available_zero_cols = [col for col in zero_replace_columns if col in df_copy.columns]
    
    if available_zero_cols:
        df_copy[available_zero_cols] = df_copy[available_zero_cols].replace(0, np.nan)
        print(f"\n✓ Replaced 0 values with NaN in: {available_zero_cols}")
    
    # Calculate preprocessing parameters (mean/median)
    preprocessing_params = {}
    
    if 'Glucose' in df_copy.columns:
        glucose_mean = df_copy['Glucose'].mean()
        preprocessing_params['glucose_mean'] = glucose_mean
        df_copy['Glucose'].fillna(glucose_mean, inplace=True)
        print(f"✓ Glucose: filled {df_copy['Glucose'].isnull().sum()} missing values with mean = {glucose_mean:.2f}")
    
    if 'BloodPressure' in df_copy.columns:
        bloodpressure_mean = df_copy['BloodPressure'].mean()
        preprocessing_params['bloodpressure_mean'] = bloodpressure_mean
        df_copy['BloodPressure'].fillna(bloodpressure_mean, inplace=True)
        print(f"✓ BloodPressure: filled missing values with mean = {bloodpressure_mean:.2f}")
    
    if 'SkinThickness' in df_copy.columns:
        skinthickness_median = df_copy['SkinThickness'].median()
        preprocessing_params['skinthickness_median'] = skinthickness_median
        df_copy['SkinThickness'].fillna(skinthickness_median, inplace=True)
        print(f"✓ SkinThickness: filled missing values with median = {skinthickness_median:.2f}")
    
    if 'Insulin' in df_copy.columns:
        insulin_median = df_copy['Insulin'].median()
        preprocessing_params['insulin_median'] = insulin_median
        df_copy['Insulin'].fillna(insulin_median, inplace=True)
        print(f"✓ Insulin: filled missing values with median = {insulin_median:.2f}")
    
    if 'BMI' in df_copy.columns:
        bmi_median = df_copy['BMI'].median()
        preprocessing_params['bmi_median'] = bmi_median
        df_copy['BMI'].fillna(bmi_median, inplace=True)
        print(f"✓ BMI: filled missing values with median = {bmi_median:.2f}")
    
    # Separate features and target
    if 'Outcome' not in df_copy.columns:
        raise ValueError("ERROR: 'Outcome' column not found in dataset!")
    
    X = df_copy.drop(columns='Outcome')
    y = df_copy['Outcome']
    
    # Store feature names
    feature_names = list(X.columns)
    
    print(f"\n✓ Final feature set ({len(feature_names)} features): {feature_names}")
    print(f"✓ Target variable 'Outcome' distribution:\n{y.value_counts()}")
    print(f"✓ Preprocessed dataset shape: {X.shape}")
    
    return X, y, feature_names, preprocessing_params

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train, model_type='random_forest', random_state=42):
    """
    Train a diabetes prediction model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: 'random_forest' or 'xgboost'
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    if model_type == 'random_forest':
        print(f"Training Random Forest Classifier...")
        print(f"  - n_estimators: {RF_N_ESTIMATORS}")
        print(f"  - max_depth: {RF_MAX_DEPTH}")
        print(f"  - min_samples_split: {RF_MIN_SAMPLES_SPLIT}")
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            random_state=random_state,
            n_jobs=-1
        )
        
    elif model_type == 'xgboost':
        print(f"Training XGBoost Classifier...")
        print(f"  - n_estimators: {XGB_N_ESTIMATORS}")
        print(f"  - max_depth: {XGB_MAX_DEPTH}")
        print(f"  - learning_rate: {XGB_LEARNING_RATE}")
        
        try:
            import xgboost as xgb
        except ImportError:
            print("ERROR: XGBoost not installed. Install with: pip install xgboost")
            raise
        
        model = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=random_state,
            eval_metric='logloss'
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'random_forest' or 'xgboost'")
    
    # Train the model
    model.fit(X_train, y_train)
    print("✓ Model training completed!")
    
    return model

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              No    Yes")
    print(f"Actual No   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Yes  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return metrics

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to train the model with XAI support."""
    
    print("\n" + "="*70)
    print("DIABETES PREDICTION MODEL WITH EXPLAINABLE AI (XAI)")
    print("="*70)
    print(f"Model Type: {MODEL_TYPE.upper()}")
    print(f"Dataset: {DATASET_PATH}")
    print("="*70)
    
    # Load and preprocess data
    X, y, feature_names, preprocessing_params = load_and_preprocess_data(DATASET_PATH)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n✓ Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Train model
    model = train_model(X_train, y_train, model_type=MODEL_TYPE, random_state=RANDOM_STATE)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model and preprocessing parameters
    print("\n" + "="*70)
    print("SAVING MODEL AND PARAMETERS")
    print("="*70)
    
    joblib.dump(model, MODEL_FILENAME)
    print(f"✓ Model saved to: {MODEL_FILENAME}")
    
    joblib.dump(preprocessing_params, PREPROCESSING_FILENAME)
    print(f"✓ Preprocessing parameters saved to: {PREPROCESSING_FILENAME}")
    
    joblib.dump(feature_names, FEATURE_NAMES_FILENAME)
    print(f"✓ Feature names saved to: {FEATURE_NAMES_FILENAME}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run 'python xai.py' to generate SHAP explanations")
    print("2. Use 'explain_prediction()' function from xai.py for individual predictions")
    print("3. Update app.py to integrate XAI explanations")
    print("="*70)
    
    return model, X_train, X_test, y_train, y_test, feature_names, preprocessing_params

if __name__ == '__main__':
    model, X_train, X_test, y_train, y_test, feature_names, preprocessing_params = main()

