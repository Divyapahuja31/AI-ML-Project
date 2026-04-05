"""
Comprehensive Training Script for Credit Risk Model.
Includes:
- Data preprocessing
- K-Fold Cross-Validation (5 folds)
- Metric logging for EACH fold
- Final model training and artifact serialization
- Training time logging
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import StratifiedKFold

# Import local modules
from src.preprocess import preprocess_pipeline

# Configuration
DATA_PATH = "data/Credit Risk Benchmark Dataset.csv"
MODEL_DIR = "models"
MODEL_NAME = "risk_model.pkl"

def main():
    print(f"🚀 Starting Training Pipeline - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    # 1. Load and Preprocess
    print("⏳ Loading data and running preprocessing pipeline...")
    X_train, X_test, y_train, y_test = preprocess_pipeline(DATA_PATH)
    feature_names = list(X_train.columns)
    
    # 2. K-Fold Cross-Validation
    print("\n🔍 Starting 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    # Standard hyperparameters
    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        fold_start = time.time()
        
        X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_f_train, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = RandomForestClassifier(**params)
        model.fit(X_f_train, y_f_train)
        
        y_pred = model.predict(X_f_val)
        y_prob = model.predict_proba(X_f_val)[:, 1]
        
        metrics = {
            "fold": fold,
            "accuracy": accuracy_score(y_f_val, y_pred),
            "precision": precision_score(y_f_val, y_pred),
            "recall": recall_score(y_f_val, y_pred),
            "f1": f1_score(y_f_val, y_pred),
            "roc_auc": roc_auc_score(y_f_val, y_prob),
            "duration": time.time() - fold_start
        }
        fold_metrics.append(metrics)
        print(f"  ✅ Fold {fold} Complete | ROC-AUC: {metrics['roc_auc']:.4f} | Time: {metrics['duration']:.2f}s")
    
    # Calculate Aggregate Metrics
    avg_roc_auc = np.mean([m['roc_auc'] for m in fold_metrics])
    std_roc_auc = np.std([m['roc_auc'] for m in fold_metrics])
    
    print(f"\n📈 CV Results: Mean ROC-AUC = {avg_roc_auc:.4f} (±{std_roc_auc:.4f})")
    
    # 3. Final Training on full train set
    print("\n⏳ Training final model on full training set...")
    final_model = RandomForestClassifier(**params)
    final_model.fit(X_train, y_train)
    
    # 4. Final Evaluation on test set
    y_test_pred = final_model.predict(X_test)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1": f1_score(y_test, y_test_pred),
        "roc_auc": roc_auc_score(y_test, y_test_prob)
    }
    
    total_duration = time.time() - start_time
    print(f"\n✨ Final Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"⏱️ Total Training Duration: {total_duration:.2f}s")
    
    # 5. Save Artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_names": feature_names,
        "metrics": test_metrics,
        "cv_metrics": fold_metrics,
        "training_metadata": {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": total_duration,
            "n_samples": len(X_train) + len(X_test),
            "cv_folds": 5
        }
    }
    
    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump(artifact, save_path)
    print(f"\n✅ Model artifact saved to {save_path}")

if __name__ == "__main__":
    main()
