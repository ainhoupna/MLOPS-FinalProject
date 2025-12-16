"""
Data preprocessing module for Credit Card Fraud Detection.
Handles data loading, validation, splitting, and feature scaling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import os


class DataPreprocessor:
    """Preprocessor for credit card fraud detection data."""

    def __init__(self, data_path: str = "data/raw/creditcard.csv"):
        """
        Initialize the preprocessor.

        Args:
            data_path: Path to the raw credit card dataset
        """
        self.data_path = data_path
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """
        Load the credit card fraud dataset.

        Returns:
            DataFrame with the loaded data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download from Kaggle: "
                "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )

        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} records")
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean() * 100:.3f}%)")

        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the dataset structure and content.

        Args:
            df: DataFrame to validate

        Returns:
            True if validation passes
        """
        # Check required columns
        required_cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
        missing_cols = set(required_cols) - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for missing values
        if df.isnull().any().any():
            raise ValueError("Dataset contains missing values")

        # Check class column
        if not set(df["Class"].unique()).issubset({0, 1}):
            raise ValueError("Class column should only contain 0 and 1")

        print("Data validation passed ✓")
        return True

    def prepare_features(
        self, df: pd.DataFrame, fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features for modeling.
        Scale Time and Amount features (V1-V28 are already PCA-transformed).

        Args:
            df: Input DataFrame
            fit_scaler: Whether to fit the scaler (True for train, False for test)

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        # Scale Time and Amount (V1-V28 are already scaled from PCA)
        if fit_scaler:
            df[["Time", "Amount"]] = self.scaler.fit_transform(df[["Time", "Amount"]])
        else:
            df[["Time", "Amount"]] = self.scaler.transform(df[["Time", "Amount"]])

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets with stratification.

        Args:
            df: Input DataFrame
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df["Class"]
        )

        # Second split: train vs val
        train, val = train_test_split(
            train_val,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val["Class"],
        )

        print("\nData split:")
        print(f"Train: {len(train)} samples ({train['Class'].sum()} frauds)")
        print(f"Val:   {len(val)} samples ({val['Class'].sum()} frauds)")
        print(f"Test:  {len(test)} samples ({test['Class'].sum()} frauds)")

        return train, val, test

    def preprocess_pipeline(
        self, save_processed: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full preprocessing pipeline: load, validate, scale, and split.

        Args:
            save_processed: Whether to save processed data to disk

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Load and validate
        df = self.load_data()
        self.validate_data(df)

        # Split first (before scaling to prevent data leakage)
        train, val, test = self.split_data(df)

        # Scale features
        train = self.prepare_features(train, fit_scaler=True)
        val = self.prepare_features(val, fit_scaler=False)
        test = self.prepare_features(test, fit_scaler=False)

        # Save processed data
        if save_processed:
            os.makedirs("data/processed", exist_ok=True)
            train.to_csv("data/processed/train.csv", index=False)
            val.to_csv("data/processed/val.csv", index=False)
            test.to_csv("data/processed/test.csv", index=False)
            print("\n✓ Processed data saved to data/processed/")

            # Save scaler
            import pickle

            os.makedirs("models", exist_ok=True)
            with open("models/scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            print("✓ Scaler saved to models/scaler.pkl")

        return train, val, test


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (X, y)
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.preprocess_pipeline()

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)

    print("\nFeature shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
