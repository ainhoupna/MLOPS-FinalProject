"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor, get_X_y


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample fraud detection data."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Time': np.random.randint(0, 172800, n_samples),
            'Amount': np.random.uniform(0, 1000, n_samples),
            'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        }
        
        # Add V1-V28 features
        for i in range(1, 29):
            data[f'V{i}'] = np.random.randn(n_samples)
        
        return pd.DataFrame(data)
    
    def test_validate_data_success(self, sample_data):
        """Test data validation with valid data."""
        preprocessor = DataPreprocessor()
        assert preprocessor.validate_data(sample_data) == True
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        df = pd.DataFrame({'Time': [1, 2, 3], 'Amount': [10, 20, 30]})
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocessor.validate_data(df)
    
    def test_validate_data_missing_values(self, sample_data):
        """Test data validation with missing values."""
        sample_data.loc[0, 'Amount'] = np.nan
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="contains missing values"):
            preprocessor.validate_data(sample_data)
    
    def test_validate_data_invalid_class(self, sample_data):
        """Test data validation with invalid class values."""
        sample_data.loc[0, 'Class'] = 2
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="should only contain 0 and 1"):
            preprocessor.validate_data(sample_data)
    
    def test_split_data_stratification(self, sample_data):
        """Test that data splitting maintains class distribution."""
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.split_data(sample_data)
        
        # Check sizes
        assert len(train) + len(val) + len(test) == len(sample_data)
        
        # Check stratification (fraud rate should be similar)
        original_fraud_rate = sample_data['Class'].mean()
        train_fraud_rate = train['Class'].mean()
        val_fraud_rate = val['Class'].mean()
        test_fraud_rate = test['Class'].mean()
        
        # Allow some tolerance due to small sample size
        assert abs(train_fraud_rate - original_fraud_rate) < 0.01
        assert abs(test_fraud_rate - original_fraud_rate) < 0.01
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation and scaling."""
        preprocessor = DataPreprocessor()
        
        # Fit scaler on training data
        scaled_data = preprocessor.prepare_features(sample_data, fit_scaler=True)
        
        # Check that Time and Amount are scaled
        assert scaled_data['Time'].mean() < 1.0  # Should be normalized
        assert scaled_data['Amount'].mean() < 1.0
        
        # Check that all columns are present
        assert len(scaled_data.columns) == len(sample_data.columns)
    
    def test_get_X_y(self, sample_data):
        """Test feature-target separation."""
        X, y = get_X_y(sample_data)
        
        # Check shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        
        # Check that Class is not in features
        assert 'Class' not in X.columns
        
        # Check that y contains only 0 and 1
        assert set(y.unique()).issubset({0, 1})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
