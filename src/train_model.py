import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.preprocess import preprocess_pipeline


DATA_PATH = "data/Credit Risk Benchmark Dataset.csv"
MODEL_PATH = "models/risk_model.pkl"


def train_model():
    X_train, X_test, y_train, y_test = preprocess_pipeline(DATA_PATH)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


if __name__ == "__main__":
    train_model() 