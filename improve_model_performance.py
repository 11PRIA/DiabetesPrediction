"""
Model Performance Improvement Guide

This script provides comprehensive strategies and code examples to improve
your diabetes prediction model's accuracy, precision, recall, F1-score, and AUC-ROC.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STRATEGY 1: HYPERPARAMETER TUNING
# ============================================================================

def hyperparameter_tuning_random_forest(X_train, y_train, cv=5, scoring='roc_auc'):
    """
    Perform hyperparameter tuning for Random Forest using Grid Search.
    
    This will find the best combination of hyperparameters to maximize performance.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='roc_auc'
        Metric to optimize
        
    Returns:
    --------
    best_model : RandomForestClassifier
        Best model with tuned hyperparameters
    best_params : dict
        Best hyperparameters found
    """
    
    print("\n" + "="*80)
    print("STRATEGY 1: HYPERPARAMETER TUNING")
    print("="*80)
    
    # Define parameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300, 500],  # Number of trees
        'max_depth': [10, 20, 30, None],       # Maximum depth of trees
        'min_samples_split': [2, 5, 10],       # Minimum samples to split
        'min_samples_leaf': [1, 2, 4],         # Minimum samples in leaf
        'max_features': ['sqrt', 'log2', None], # Features to consider
        'class_weight': [None, 'balanced']     # Handle class imbalance
    }
    
    # Create base model
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform Grid Search with cross-validation
    print("\nPerforming Grid Search with Cross-Validation...")
    print("This may take several minutes...")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n✓ Best Cross-Validation Score ({scoring}): {best_score:.4f}")
    print(f"\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_model, best_params


def hyperparameter_tuning_randomized(X_train, y_train, n_iter=50, cv=5):
    """
    Faster alternative: Randomized Search for hyperparameter tuning.
    
    Tests random combinations instead of all combinations (faster).
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    n_iter : int, default=50
        Number of random combinations to try
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    best_model : RandomForestClassifier
        Best model found
    """
    
    print("\n" + "="*80)
    print("STRATEGY 1B: RANDOMIZED HYPERPARAMETER SEARCH (FASTER)")
    print("="*80)
    
    # Define parameter distributions (for random sampling)
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [5, 10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced']
    }
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("\nPerforming Randomized Search...")
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"\n✓ Best Score: {random_search.best_score_:.4f}")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_model, best_params


# ============================================================================
# STRATEGY 2: FEATURE ENGINEERING
# ============================================================================

def create_new_features(df):
    """
    Create new engineered features that may improve model performance.
    
    Parameters:
    -----------
    df : DataFrame
        Original dataset
        
    Returns:
    --------
    df_enhanced : DataFrame
        Dataset with new features added
    """
    
    print("\n" + "="*80)
    print("STRATEGY 2: FEATURE ENGINEERING")
    print("="*80)
    
    df_enhanced = df.copy()
    
    # 1. BMI Categories (may be more informative than raw BMI)
    df_enhanced['BMI_Category'] = pd.cut(
        df_enhanced['BMI'],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]  # Underweight, Normal, Overweight, Obese
    ).astype(float)
    
    # 2. Glucose Categories
    df_enhanced['Glucose_Category'] = pd.cut(
        df_enhanced['Glucose'],
        bins=[0, 100, 125, 200],
        labels=[0, 1, 2]  # Normal, Prediabetic, Diabetic
    ).astype(float)
    
    # 3. Age Groups
    df_enhanced['Age_Group'] = pd.cut(
        df_enhanced['Age'],
        bins=[0, 30, 45, 60, 100],
        labels=[0, 1, 2, 3]  # Young, Middle-aged, Senior, Elderly
    ).astype(float)
    
    # 4. Interaction Features (combinations of important features)
    df_enhanced['Glucose_BMI'] = df_enhanced['Glucose'] * df_enhanced['BMI']
    df_enhanced['Age_BMI'] = df_enhanced['Age'] * df_enhanced['BMI']
    df_enhanced['Glucose_Insulin'] = df_enhanced['Glucose'] * df_enhanced['Insulin']
    
    # 5. Ratio Features
    df_enhanced['Glucose_Insulin_Ratio'] = df_enhanced['Glucose'] / (df_enhanced['Insulin'] + 1)
    df_enhanced['BMI_BloodPressure_Ratio'] = df_enhanced['BMI'] / (df_enhanced['BloodPressure'] + 1)
    
    # 6. Polynomial Features (squared terms for non-linear relationships)
    df_enhanced['BMI_Squared'] = df_enhanced['BMI'] ** 2
    df_enhanced['Glucose_Squared'] = df_enhanced['Glucose'] ** 2
    df_enhanced['Age_Squared'] = df_enhanced['Age'] ** 2
    
    print("✓ Created new features:")
    print("  - BMI_Category, Glucose_Category, Age_Group")
    print("  - Interaction features (Glucose_BMI, Age_BMI, Glucose_Insulin)")
    print("  - Ratio features (Glucose_Insulin_Ratio, BMI_BloodPressure_Ratio)")
    print("  - Polynomial features (BMI_Squared, Glucose_Squared, Age_Squared)")
    
    return df_enhanced


# ============================================================================
# STRATEGY 3: FEATURE SELECTION
# ============================================================================

def select_best_features(X_train, y_train, X_test, method='mutual_info', k=10):
    """
    Select the most important features to reduce overfitting and improve performance.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    method : str, default='mutual_info'
        'mutual_info' or 'f_classif'
    k : int, default=10
        Number of top features to select
        
    Returns:
    --------
    X_train_selected : array-like
        Selected training features
    X_test_selected : array-like
        Selected test features
    selector : SelectKBest
        Fitted selector object
    """
    
    print("\n" + "="*80)
    print("STRATEGY 3: FEATURE SELECTION")
    print("="*80)
    
    # Choose scoring function
    if method == 'mutual_info':
        score_func = mutual_info_classif
    else:
        score_func = f_classif
    
    # Create selector
    selector = SelectKBest(score_func=score_func, k=k)
    
    # Fit and transform
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names (if available)
    if hasattr(X_train, 'columns'):
        selected_features = X_train.columns[selector.get_support()].tolist()
        print(f"\n✓ Selected top {k} features:")
        for i, feat in enumerate(selected_features, 1):
            print(f"  {i}. {feat}")
    else:
        print(f"\n✓ Selected top {k} features (indices: {selector.get_support(indices=True)})")
    
    return X_train_selected, X_test_selected, selector


# ============================================================================
# STRATEGY 4: HANDLE CLASS IMBALANCE
# ============================================================================

def handle_class_imbalance(X_train, y_train, method='smote'):
    """
    Handle class imbalance using various techniques.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    method : str, default='smote'
        'smote', 'undersample', 'oversample', or 'class_weight'
        
    Returns:
    --------
    X_balanced : array-like
        Balanced training features
    y_balanced : array-like
        Balanced training labels
    """
    
    print("\n" + "="*80)
    print("STRATEGY 4: HANDLE CLASS IMBALANCE")
    print("="*80)
    
    from collections import Counter
    print(f"\nOriginal class distribution: {Counter(y_train)}")
    
    if method == 'class_weight':
        # Use class_weight parameter in model (no data modification needed)
        print("\n✓ Using class_weight='balanced' in model")
        return X_train, y_train
    
    elif method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            print(f"\n✓ SMOTE applied. New distribution: {Counter(y_balanced)}")
            return X_balanced, y_balanced
        except ImportError:
            print("\n⚠ SMOTE not available. Install: pip install imbalanced-learn")
            return X_train, y_train
    
    elif method == 'oversample':
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = ros.fit_resample(X_train, y_train)
        print(f"\n✓ Random Oversampling applied. New distribution: {Counter(y_balanced)}")
        return X_balanced, y_balanced
    
    elif method == 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X_train, y_train)
        print(f"\n✓ Random Undersampling applied. New distribution: {Counter(y_balanced)}")
        return X_balanced, y_balanced
    
    else:
        return X_train, y_train


# ===========================================================================
# STRATEGY 5: DATA PREPROCESSING IMPROVEMENTS
# ============================================================================

def improved_preprocessing(df):
    """
    Enhanced preprocessing with better handling of outliers and scaling.
    
    Parameters:
    -----------
    df : DataFrame
        Raw dataset
        
    Returns:
    --------
    df_processed : DataFrame
        Preprocessed dataset
    """
    
    print("\n" + "="*80)
    print("STRATEGY 5: IMPROVED PREPROCESSING")
    print("="*80)
    
    df_processed = df.copy()
    
    # Handle missing values (0s) with better imputation
    zero_replace_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_replace_columns:
        if col in df_processed.columns:
            # Replace 0 with NaN
            df_processed[col] = df_processed[col].replace(0, np.nan)
            
            # Use median for skewed distributions, mean for normal
            if col in ['Glucose', 'BloodPressure']:
                fill_value = df_processed[col].median()  # More robust to outliers
            else:
                fill_value = df_processed[col].median()
            
            df_processed[col].fillna(fill_value, inplace=True)
    
    # Handle outliers using IQR method (optional)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Outcome':
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
    
    print("✓ Applied improved preprocessing:")
    print("  - Better imputation strategy")
    print("  - Outlier capping using IQR method")
    
    return df_processed


# ============================================================================
# STRATEGY 6: ENSEMBLE METHODS
# ============================================================================

def create_ensemble_model(X_train, y_train):
    """
    Create an ensemble model combining multiple algorithms.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
        
    Returns:
    --------
    ensemble_model : VotingClassifier
        Ensemble model
    """
    
    print("\n" + "="*80)
    print("STRATEGY 6: ENSEMBLE METHODS")
    print("="*80)
    
    # Create multiple models
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Create voting ensemble (soft voting uses probabilities)
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft',  # Use probability voting
        n_jobs=-1
    )
    
    print("✓ Created ensemble model:")
    print("  - Random Forest (200 trees)")
    print("  - Gradient Boosting (100 trees)")
    print("  - Soft voting (probability-based)")
    
    ensemble_model.fit(X_train, y_train)
    
    return ensemble_model


# ============================================================================
# STRATEGY 7: CROSS-VALIDATION FOR ROBUST EVALUATION
# ============================================================================

def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Perform cross-validation to get robust performance estimates.
    
    Parameters:
    -----------
    model : sklearn model
        Model to evaluate
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    cv : int, default=5
        Number of folds
        
    Returns:
    --------
    cv_results : dict
        Cross-validation results
    """
    
    print("\n" + "="*80)
    print("STRATEGY 7: CROSS-VALIDATION")
    print("="*80)
    
    # Use stratified K-fold for imbalanced datasets
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Evaluate multiple metrics
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = {}
    
    for metric_name, metric_scorer in scoring_metrics.items():
        scores = cross_val_score(model, X_train, y_train, 
                                cv=cv_strategy, scoring=metric_scorer, n_jobs=-1)
        cv_results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_results


# ============================================================================
# COMPREHENSIVE IMPROVEMENT PIPELINE
# ============================================================================

def comprehensive_improvement_pipeline(dataset_path='kaggle_diabetes.csv'):
    """
    Complete pipeline applying all improvement strategies.
    
    This function demonstrates how to combine all strategies for maximum improvement.
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL IMPROVEMENT PIPELINE")
    print("="*80)
    
    # Load data
    df = pd.read_csv(dataset_path)
    if 'DiabetesPedigreeFunction' in df.columns:
        df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
    
    # Step 1: Improved preprocessing
    df_processed = improved_preprocessing(df)
    
    # Step 2: Feature engineering
    df_enhanced = create_new_features(df_processed)
    
    # Prepare features and target
    X = df_enhanced.drop(columns='Outcome')
    y = df_enhanced['Outcome']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 3: Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(
        X_train, y_train, method='class_weight'
    )
    
    # Step 4: Feature selection (optional - comment out if you want all features)
    # X_train_selected, X_test_selected, selector = select_best_features(
    #     X_train_balanced, y_train_balanced, X_test, k=15
    # )
    # X_train_final, X_test_final = X_train_selected, X_test_selected
    X_train_final, X_test_final = X_train_balanced, X_test
    
    # Step 5: Hyperparameter tuning
    print("\n" + "="*80)
    print("Training optimized model...")
    print("="*80)
    
    # Use RandomizedSearch for faster results (or GridSearch for thorough search)
    best_model, best_params = hyperparameter_tuning_randomized(
        X_train_final, y_train_balanced, n_iter=30, cv=5
    )
    
    # Step 6: Cross-validation
    cv_results = cross_validate_model(best_model, X_train_final, y_train_balanced, cv=5)
    
    # Step 7: Final evaluation on test set
    from evaluate_model import evaluate_diabetes_model
    final_results = evaluate_diabetes_model(best_model, X_test_final, y_test)
    
    return best_model, final_results, best_params


# ============================================================================
# QUICK IMPROVEMENTS (MINIMAL CODE CHANGES)
# ============================================================================

def quick_improvements_to_model_py():
    """
    Show quick improvements you can make to model.py with minimal changes.
    """
    
    improvements = """
    # QUICK IMPROVEMENTS FOR model.py:
    
    # 1. Increase n_estimators (more trees = better performance, but slower)
    RF_N_ESTIMATORS = 200  # Change from 100 to 200-500
    
    # 2. Add class_weight to handle imbalance
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',  # ADD THIS LINE
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Add more hyperparameters
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,              # Limit depth to prevent overfitting
        min_samples_split=5,      # Require more samples to split
        min_samples_leaf=2,       # ADD: Minimum samples in leaf
        max_features='sqrt',      # ADD: Limit features per split
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Use XGBoost (often performs better than Random Forest)
    MODEL_TYPE = 'xgboost'  # Change from 'random_forest'
    XGB_N_ESTIMATORS = 200
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.05  # Lower learning rate with more trees
    
    # 5. Add cross-validation during training
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    """
    
    print("\n" + "="*80)
    print("QUICK IMPROVEMENTS (Copy to model.py)")
    print("="*80)
    print(improvements)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    """
    Run comprehensive improvement pipeline or individual strategies.
    """
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE IMPROVEMENT GUIDE")
    print("="*80)
    print("\nThis script provides multiple strategies to improve your model.")
    print("\nChoose an option:")
    print("1. Run comprehensive improvement pipeline (all strategies)")
    print("2. Show quick improvements for model.py")
    print("3. Individual strategy examples")
    
    # Show quick improvements
    quick_improvements_to_model_py()
    
    # Uncomment to run comprehensive pipeline:
    # best_model, results, params = comprehensive_improvement_pipeline()

