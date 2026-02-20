import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def encode_data(df):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def separate_features_target(df, target_col="dlq_2yrs"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(filepath, target_col="dlq_2yrs"):
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_data(df)
    X, y = separate_features_target(df, target_col)
    return split_data(X, y)