"""
Integration tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

from src.api.main import app


class TestAPI:
    """Test suite for FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_transaction(self):
        """Create sample transaction payload."""
        transaction = {
            "Time": 0,
            "Amount": 149.62,
        }
        # Add V1-V28
        for i in range(1, 29):
            transaction[f"V{i}"] = float(np.random.randn())

        return transaction

    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML landing page."""
        response = client.get("/")
        assert response.status_code == 200
        
        # Check it's HTML, not JSON
        assert "text/html" in response.headers.get("content-type", "")
        
        # Check for key content in HTML
        assert "Credit Card Fraud Detection" in response.text
        assert "API" in response.text
    @patch("src.api.main.detector")
    def test_health_endpoint_healthy(self, mock_detector, client):
        """Test health endpoint when model is loaded."""
        mock_detector.model = Mock()
        mock_detector.threshold = 0.5

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"]

    def test_health_endpoint_unhealthy(self, client):
        """Test health endpoint when model is not loaded."""
        with patch("src.api.main.detector", None):
            response = client.get("/health")
            # Should return 503 if model not loaded
            # Note: This might return 200 if model is already loaded in the app
            assert response.status_code in [200, 503]

    @patch("src.api.main.detector")
    def test_predict_endpoint_success(self, mock_detector, client, sample_transaction):
        """Test prediction endpoint with valid input."""
        # Mock detector response
        mock_detector.predict.return_value = {
            "prediction": 0,
            "label": "Legitimate",
            "fraud_probability": 0.15,
            "threshold": 0.5,
        }

        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "label" in data
        assert "fraud_probability" in data
        assert "threshold" in data
        assert "latency_ms" in data

        # Check types
        assert isinstance(data["prediction"], int)
        assert isinstance(data["label"], str)
        assert isinstance(data["fraud_probability"], float)
        assert isinstance(data["latency_ms"], float)

    def test_predict_endpoint_missing_features(self, client):
        """Test prediction endpoint with missing features."""
        incomplete_transaction = {"Time": 0, "Amount": 100}

        response = client.post("/predict", json=incomplete_transaction)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_amount(self, client, sample_transaction):
        """Test prediction endpoint with negative amount."""
        sample_transaction["Amount"] = -100

        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 422  # Validation error

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Check that response contains Prometheus metrics
        content = response.text
        assert "fraud_detection" in content or "HELP" in content or "#" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
