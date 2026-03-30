"""
Unit tests for the preprocessing module.
Ensures data cleaning, imputation, and splitting logic remains consistent.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_data, separate_features_target, TARGET_COL, FEATURE_COLS

class TestPreprocess(unittest.TestCase):

    def setUp(self):
        """Create a dirty synthetic dataset for testing."""
        self.data = pd.DataFrame({
            "rev_util":    [0.1, 0.5, np.nan, 10.0, 0.2],
            "age":         [25, 45, 30, 80, 50],
            "late_30_59":  [0, 1, 0, 5, 0],
            "debt_ratio":  [0.3, 0.4, 0.5, 0.6, 0.7],
            "monthly_inc": [5000, 6000, 7000, 8000, np.nan],
            "open_credit": [5, 10, 8, 15, 6],
            "late_90":     [0, 0, 1, 0, 0],
            "real_estate": [0, 1, 0, 2, 1],
            "late_60_89":  [0, 0, 0, 0, 0],
            "dependents":  [0, 1, 2, 0, 1],
            TARGET_COL:    [0, 1, 0, 1, 0]
        })

    def test_clean_data_imputation(self):
        """Test median imputation for missing values."""
        cleaned = clean_data(self.data)
        self.assertFalse(cleaned["monthly_inc"].isnull().any())
        self.assertFalse(cleaned["rev_util"].isnull().any())
        # Monthly income median for first 4 is (5000+8000)/2 = 6500 or mean? 
        # Actually median of [5k, 6k, 7k, 8k] is 6500.
        self.assertEqual(cleaned["monthly_inc"].iloc[4], 6500)

    def test_clean_data_capping(self):
        """Test outlier capping logic."""
        # Create a df with an obvious outlier for monthly_inc
        df = self.data.copy()
        df.loc[0, "monthly_inc"] = 1_000_000 # huge outlier
        cleaned = clean_data(df)
        cap = df["monthly_inc"].quantile(0.99)
        self.assertLessEqual(cleaned["monthly_inc"].max(), cap)

    def test_separate_features_target(self):
        """Test separation of X and y."""
        X, y = separate_features_target(self.data)
        self.assertEqual(len(X.columns), len(FEATURE_COLS))
        self.assertEqual(len(y), len(self.data))
        self.assertNotIn(TARGET_COL, X.columns)

    def test_missing_target_removal(self):
        """Test that rows with missing target labels are dropped."""
        df = self.data.copy()
        df.loc[0, TARGET_COL] = np.nan
        cleaned = clean_data(df)
        self.assertEqual(len(cleaned), len(df) - 1)

if __name__ == "__main__":
    unittest.main()
