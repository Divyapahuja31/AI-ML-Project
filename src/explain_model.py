import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
from src.preprocess import preprocess_pipeline

MODEL_PATH = "models/risk_model.pkl"
DATA_PATH = "data/Credit Risk Benchmark Dataset.csv"

def get_feature_importance():
    model = joblib.load(MODEL_PATH)

    X_train, X_test, y_train, y_test = preprocess_pipeline(DATA_PATH)
    X = X_train


    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print(feature_importance.head(10))

    return feature_importance


if __name__ == "__main__":
    get_feature_importance()