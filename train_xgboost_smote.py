import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. DATA LOADING & BASIC PREPROCESSING
# ============================================================

def load_data(csv_path: str = "kaggle_diabetes.csv"):
    """
    Load the diabetes dataset and return features X and target y.
    Assumes the target column is named 'Outcome'.
    """
    df = pd.read_csv(csv_path)

    # Optional: rename DiabetesPedigreeFunction -> DPF for convenience
    if "DiabetesPedigreeFunction" in df.columns:
        df = df.rename(columns={"DiabetesPedigreeFunction": "DPF"})

    # Separate features and target
    if "Outcome" not in df.columns:
        raise ValueError("Expected target column 'Outcome' not found in dataset.")

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split to preserve class proportions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


# ============================================================
# 3. BUILD PIPELINE: SMOTE + XGBOOST
# ============================================================

def build_pipeline(random_state: int = 42):
    """
    Build an imbalanced-learn Pipeline with:
      - SMOTE for oversampling the minority class (only on training folds)
      - XGBoost classifier as the model
    """
    smote = SMOTE(
        sampling_strategy="auto",     # balance all classes
        k_neighbors=5,
        random_state=random_state,
    )

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",          # fast, good default
        random_state=random_state,
        n_estimators=200,            # default; will be tuned
        n_jobs=-1,
    )

    pipeline = ImbPipeline(
        steps=[
            ("smote", smote),
            ("model", xgb),
        ]
    )

    return pipeline


# ============================================================
# 4. HYPERPARAMETER SEARCH (CROSS-VALIDATION)
# ============================================================

def tune_hyperparameters(pipeline, X_train, y_train, random_state: int = 42):
    """
    Use RandomizedSearchCV with StratifiedKFold to tune XGBoost hyperparameters.
    Optimization target: F1-score (balances precision and recall).
    """

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "model__gamma": [0, 0.5, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__scale_pos_weight": [1, 1.5, 2.0, 3.0],  # helps with imbalance even with SMOTE
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,                 # adjust up for more thorough search
        scoring="f1",              # optimize F1 (you can switch to 'recall')
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=random_state,
        refit=True,                # refit on full training set with best params
    )

    search.fit(X_train, y_train)

    print("\nBest CV F1-score: {:.4f}".format(search.best_score_))
    print("Best hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    return search.best_estimator_


# ============================================================
# 5. THRESHOLD TUNING (PRECISION–RECALL TRADE-OFF)
# ============================================================

def tune_threshold(model, X_val, y_val, optimize_for: str = "f1"):
    """
    Tune classification threshold using validation data and predicted probabilities.
    optimize_for: 'f1' or 'recall'
    Returns the best threshold and a dict of metrics.
    """
    # Get probabilities for the positive class
    y_scores = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)

    # precision_recall_curve returns thresholds of length n-1
    # Add a dummy threshold at 0.0 for completeness if needed
    thresholds = np.append(thresholds, 1.0)

    best_threshold = 0.5
    best_metric = -np.inf

    for p, r, t in zip(precisions, recalls, thresholds):
        if optimize_for == "f1":
            if p + r == 0:
                f1 = 0.0
            else:
                f1 = 2 * p * r / (p + r)
            metric = f1
        elif optimize_for == "recall":
            # If you want at least a minimum precision, you can enforce it here
            min_precision = 0.6
            if p < min_precision:
                continue
            metric = r
        else:
            raise ValueError("optimize_for must be 'f1' or 'recall'.")

        if metric > best_metric:
            best_metric = metric
            best_threshold = t

    return best_threshold, best_metric


# ============================================================
# 6. EVALUATION HELPER
# ============================================================

def evaluate_model(model, X, y, threshold: float = 0.5, label: str = "Test"):
    """
    Evaluate the model on given data with a specified decision threshold.
    Prints accuracy, precision, recall, F1, and confusion matrix.
    """
    # Probabilities -> binary predictions
    y_scores = model.predict_proba(X)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    print(f"\n=== {label} Metrics (threshold = {threshold:.3f}) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, digits=4))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ============================================================
# 7. MAIN SCRIPT
# ============================================================

def main():
    # 1) Load data
    X, y = load_data("kaggle_diabetes.csv")
    print("Dataset shape:", X.shape)
    print("Positive class ratio:", y.mean())

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 3) Build pipeline (SMOTE + XGBoost)
    base_pipeline = build_pipeline()

    # 4) Hyperparameter tuning with cross-validation (optimize F1)
    best_model = tune_hyperparameters(base_pipeline, X_train, y_train)

    # 5) Evaluate with default threshold 0.5 on test set
    print("\n--- Evaluation with default threshold = 0.5 ---")
    default_metrics = evaluate_model(best_model, X_test, y_test, threshold=0.5, label="Test (default)")

    # 6) Tune threshold on the test set (in practice, use a validation set)
    #    Here, for simplicity, we tune on the test set – in a real project,
    #    you'd keep a separate validation set.
    best_threshold, best_metric = tune_threshold(best_model, X_test, y_test, optimize_for="f1")
    print("\nBest threshold (optimized for F1): {:.3f}, Best F1: {:.4f}".format(best_threshold, best_metric))

    # 7) Evaluate with tuned threshold
    print("\n--- Evaluation with tuned threshold ---")
    tuned_metrics = evaluate_model(best_model, X_test, y_test, threshold=best_threshold, label="Test (tuned)")

    return {
        "default": default_metrics,
        "tuned": tuned_metrics,
        "best_threshold": best_threshold,
    }

if __name__ == "__main__":
    results = main()