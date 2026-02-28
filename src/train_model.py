"""
Model training script for credit risk classification.

Trains an optimised Random Forest, evaluates it on held-out data using
5-fold cross-validation, then serialises a *model artifact* dict that
bundles the fitted model together with its feature names and evaluation
metrics for consistent downstream use.

Usage
-----
    python3 src/train_model.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from src.preprocess import preprocess_pipeline

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/Credit Risk Benchmark Dataset.csv"
MODEL_PATH = "models/risk_model.pkl"


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model() -> dict:
    """Train a Random Forest classifier and save the model artifact.

    Returns
    -------
    dict
        Artifact containing ``model``, ``feature_names``, ``metrics``, and
        ``model_type`` keys.
    """
    print("⏳  Loading and preprocessing data …")
    X_train, X_test, y_train, y_test = preprocess_pipeline(DATA_PATH)

    print(f"    Training samples : {len(X_train):,}")
    print(f"    Test samples     : {len(X_test):,}")
    print(f"    Class balance    : {y_train.mean():.2%} positive (default)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",   # handle class imbalance
        random_state=42,
        n_jobs=-1,
    )

    print("\n⏳  Training Random Forest (n_estimators=200) …")
    model.fit(X_train, y_train)

    # ── Held-out evaluation ───────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
        "f1":        round(f1_score(y_test, y_pred),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob),   4),
    }

    # 5-fold cross-validation on training set
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
    )
    metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_roc_auc_std"]  = round(float(cv_scores.std()),  4)

    # Print summary
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<22s}: {v}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")

    # ── Persist model artifact ────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    artifact = {
        "model":         model,
        "feature_names": list(X_train.columns),
        "metrics":       metrics,
        "model_type":    "RandomForestClassifier",
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n✅  Model artifact saved → '{MODEL_PATH}'")

    return artifact


if __name__ == "__main__":
    train_model()