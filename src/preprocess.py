"""
Data preprocessing pipeline for credit risk modelling.

Handles loading, cleaning, encoding, and splitting of borrower data.
The :data:`FEATURE_COLS` constant is the single source of truth for
the model's input feature names and ordering.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Feature registry ────────────────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    "rev_util",     # Revolving utilisation of unsecured lines
    "age",          # Borrower age (years)
    "late_30_59",   # Times 30–59 days past due (last 2 yrs)
    "debt_ratio",   # Monthly debt payments / gross income
    "monthly_inc",  # Gross monthly income ($)
    "open_credit",  # Number of open credit lines
    "late_90",      # Times 90+ days past due (last 2 yrs)
    "real_estate",  # Number of real-estate loans or lines
    "late_60_89",   # Times 60–89 days past due (last 2 yrs)
    "dependents",   # Number of financial dependents
]

TARGET_COL: str = "dlq_2yrs"

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "rev_util":    "Revolving Utilization",
    "age":         "Age",
    "late_30_59":  "30–59 Days Late",
    "debt_ratio":  "Debt Ratio",
    "monthly_inc": "Monthly Income",
    "open_credit": "Open Credit Lines",
    "late_90":     "90+ Days Late",
    "real_estate": "Real Estate Loans",
    "late_60_89":  "60–89 Days Late",
    "dependents":  "Dependents",
}

# Features with heavy right-skew — cap at 99th percentile
_SKEWED_FEATURES: list[str] = ["monthly_inc", "rev_util", "debt_ratio"]


# ─── Pipeline steps ───────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load borrower CSV from *filepath*."""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data.

    Steps
    -----
    1. Drop rows where the target label is missing.
    2. Median-impute remaining NaN values.
    3. Cap extreme outliers at the 99th percentile for skewed features.
    """
    df = df.dropna(subset=[TARGET_COL]).copy()

    # Median imputation
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Outlier capping
    for col in _SKEWED_FEATURES:
        if col in df.columns:
            cap = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap)

    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode any remaining categorical / object columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def separate_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) using only the columns listed in :data:`FEATURE_COLS`."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df[target_col]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
) -> tuple:
    """Stratified 80/20 train-test split."""
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def preprocess_pipeline(
    filepath: str,
    target_col: str = TARGET_COL,
) -> tuple:
    """
    Full data transformation pipeline.

    Execution Flow:
    1. Load: Reads raw CSV from local path.
    2. Clean: Drops rows with missing targets, imputes medians, and clips 99th percentile outliers.
    3. Encode: Converts categorical data into binary indicators.
    4. Slice: Selects specific feature set and target variable.
    5. Split: Partitions data into 80% training and 20% test sets (stratified).

    Parameters:
    -----------
    filepath : str
        Path to the raw CSV data.
    target_col : str
        Name of the binary target variable (default: dlq_2yrs).

    Returns:
    --------
    tuple (X_train, X_test, y_train, y_test)
        Preprocessed training and verification sets.
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_data(df)
    X, y = separate_features_target(df, target_col)
    return split_data(X, y)