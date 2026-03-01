"""
Explainable AI module for the credit risk model.

Provides two levels of explanation:

1. **Global** – Random Forest ``feature_importances_`` sorted by magnitude.
2. **Local**  – Per-instance SHAP values via ``shap.TreeExplainer``.
   Falls back to global importances if the ``shap`` package is missing.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/risk_model.pkl"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_artifact() -> tuple:
    """Return ``(model, feature_names)`` from the saved artifact."""
    artifact = joblib.load(MODEL_PATH)
    # Backwards-compatible with old pkl format (bare model object)
    if hasattr(artifact, "predict"):
        from src.preprocess import FEATURE_COLS
        return artifact, FEATURE_COLS
    return artifact["model"], artifact["feature_names"]


# ─── Global explanation ───────────────────────────────────────────────────────

def get_global_importance() -> pd.DataFrame:
    """Return a DataFrame of global feature importances, sorted descending.

    Columns: ``feature``, ``importance``
    """
    model, feature_names = load_artifact()
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ─── Local (per-instance) explanation ─────────────────────────────────────────

def get_shap_values(input_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-instance SHAP values for a single borrower row.

    Parameters
    ----------
    input_df : pd.DataFrame
        A single-row DataFrame aligned to the model's feature space.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``shap_value``, ``abs_shap`` — sorted by
        ``abs_shap`` descending so the most influential features appear first.
    """
    model, feature_names = load_artifact()
    input_aligned = input_df[feature_names]

    try:
        import shap

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_aligned)

        # RandomForest with binary target returns a list of 2 arrays
        if isinstance(shap_values, list):
            vals = shap_values[1][0]   # Class 1 = High Risk
        else:
            vals = shap_values[0]

        result = pd.DataFrame(
            {
                "feature":    feature_names,
                "shap_value": vals,
                "abs_shap":   np.abs(vals),
            }
        ).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    except Exception:
        # Graceful fallback: use global importances as unsigned attribution
        global_imp = get_global_importance()
        result = global_imp.rename(columns={"importance": "abs_shap"})
        result["shap_value"] = result["abs_shap"]

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = get_global_importance()
    print(df.head(10).to_string(index=False))