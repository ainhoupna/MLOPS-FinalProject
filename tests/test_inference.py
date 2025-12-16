"""
Unit tests for model inference module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.models.inference import FraudDetector


class TestFraudDetector:
    """Test suite for FraudDetector class."""

    @pytest.fixture
    def sample_transaction(self):
        """Create sample transaction features."""
        transaction = {
            "Time": 0,
            "Amount": 149.62,
        }
        # Add V1-V28
        for i in range(1, 29):
            transaction[f"V{i}"] = np.random.randn()

        return transaction

    def test_validate_input_success(self, sample_transaction):
        """Test input validation with valid features."""
        detector = FraudDetector.__new__(FraudDetector)
        assert detector.validate_input(sample_transaction)

    def test_validate_input_missing_features(self):
        """Test input validation with missing features."""
        detector = FraudDetector.__new__(FraudDetector)
        incomplete_transaction = {"Time": 0, "Amount": 100}

        with pytest.raises(ValueError, match="Missing required features"):
            detector.validate_input(incomplete_transaction)

    def test_preprocess_features(self, sample_transaction):
        """Test feature preprocessing."""
        detector = FraudDetector.__new__(FraudDetector)
        detector.scaler = None

        df = detector.preprocess_features(sample_transaction)

        # Check shape
        assert df.shape == (1, 30)  # 1 row, 30 features

        # Check column order
        expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        assert list(df.columns) == expected_cols

    @patch("src.models.inference.xgb.XGBClassifier")
    def test_predict_format(self, mock_xgb, sample_transaction):
        """Test prediction output format."""
        # Mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])

        detector = FraudDetector.__new__(FraudDetector)
        detector.model = mock_model
        detector.threshold = 0.5
        detector.scaler = None

        result = detector.predict(sample_transaction)

        # Check result structure
        assert "prediction" in result
        assert "label" in result
        assert "fraud_probability" in result
        assert "threshold" in result

        # Check types
        assert isinstance(result["prediction"], int)
        assert isinstance(result["label"], str)
        assert isinstance(result["fraud_probability"], float)

        # Check values
        assert result["prediction"] in [0, 1]
        assert result["label"] in ["Fraud", "Legitimate"]
        assert 0 <= result["fraud_probability"] <= 1

    @patch("src.models.inference.xgb.XGBClassifier")
    def test_predict_threshold_application(self, mock_xgb, sample_transaction):
        """Test that threshold is correctly applied."""
        mock_model = Mock()

        detector = FraudDetector.__new__(FraudDetector)
        detector.model = mock_model
        detector.scaler = None

        # Test with probability below threshold
        detector.threshold = 0.5
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        result = detector.predict(sample_transaction)
        assert result["prediction"] == 0
        assert result["label"] == "Legitimate"

        # Test with probability above threshold
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        result = detector.predict(sample_transaction)
        assert result["prediction"] == 1
        assert result["label"] == "Fraud"

    @patch("src.models.inference.xgb.XGBClassifier")
    def test_predict_batch(self, mock_xgb, sample_transaction):
        """Test batch prediction."""
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])

        detector = FraudDetector.__new__(FraudDetector)
        detector.model = mock_model
        detector.threshold = 0.5
        detector.scaler = None

        # Create batch of transactions
        transactions = [sample_transaction, sample_transaction.copy()]

        results = detector.predict_batch(transactions)

        assert len(results) == 2
        assert all("prediction" in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
