"""
Model inference module for credit card fraud detection.
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Union
import mlflow


class FraudDetector:
    """Fraud detection inference class."""
    
    def __init__(
        self,
        model_path: str = "models/fraud_detector.json",
        threshold_path: str = "models/optimal_threshold.pkl",
        scaler_path: str = "models/scaler.pkl"
    ):
        """
        Initialize the fraud detector.
        
        Args:
            model_path: Path to the trained XGBoost model
            threshold_path: Path to the optimal threshold
            scaler_path: Path to the fitted scaler
        """
        self.model_path = model_path
        self.threshold_path = threshold_path
        self.scaler_path = scaler_path
        
        self.model = None
        self.threshold = 0.5
        self.scaler = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model, threshold, and scaler."""
        # Load XGBoost model
        if os.path.exists(self.model_path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load threshold
        if os.path.exists(self.threshold_path):
            with open(self.threshold_path, "rb") as f:
                self.threshold = pickle.load(f)
            print(f"✓ Threshold loaded: {self.threshold:.4f}")
        else:
            print(f"⚠ Threshold file not found, using default: {self.threshold}")
        
        # Load scaler
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler loaded from {self.scaler_path}")
        else:
            print(f"⚠ Scaler not found, features must be pre-scaled")
    
    def validate_input(self, features: Dict[str, float]) -> bool:
        """
        Validate input features.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            True if valid
        """
        required_features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        
        missing = set(required_features) - set(features.keys())
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        return True
    
    def preprocess_features(self, features: Dict[str, float]) -> pd.DataFrame:
        """
        Preprocess input features.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Preprocessed DataFrame
        """
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Scale Time and Amount if scaler is available
        if self.scaler is not None:
            df[['Time', 'Amount']] = self.scaler.transform(df[['Time', 'Amount']])
        
        # Ensure correct column order
        feature_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        df = df[feature_order]
        
        return df
    
    def predict(
        self,
        features: Dict[str, float],
        return_proba: bool = True
    ) -> Dict[str, Union[int, float]]:
        """
        Make fraud prediction.
        
        Args:
            features: Dictionary of transaction features
            return_proba: Whether to return probability score
            
        Returns:
            Dictionary with prediction and probability
        """
        # Validate and preprocess
        self.validate_input(features)
        X = self.preprocess_features(features)
        
        # Get probability
        proba = self.model.predict_proba(X)[0, 1]
        
        # Apply threshold
        prediction = int(proba >= self.threshold)
        
        result = {
            'prediction': prediction,
            'label': 'Fraud' if prediction == 1 else 'Legitimate',
            'fraud_probability': float(proba),
            'threshold': float(self.threshold)
        }
        
        return result
    
    def predict_batch(
        self,
        features_list: list
    ) -> list:
        """
        Make batch predictions.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for features in features_list:
            try:
                result = self.predict(features)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results


def load_from_mlflow(run_id: str, tracking_uri: str = "mlruns") -> FraudDetector:
    """
    Load model from MLFlow.
    
    Args:
        run_id: MLFlow run ID
        tracking_uri: MLFlow tracking URI
        
    Returns:
        FraudDetector instance
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    # Load model from MLFlow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)
    
    # Create detector instance
    detector = FraudDetector.__new__(FraudDetector)
    detector.model = model
    detector.threshold = 0.5  # Default, should load from artifacts
    detector.scaler = None
    
    return detector


if __name__ == "__main__":
    # Example usage
    detector = FraudDetector()
    
    # Example transaction (legitimate)
    transaction = {
        'Time': 0,
        'V1': -1.3598071336738,
        'V2': -0.0727811733098497,
        'V3': 2.53634673796914,
        'V4': 1.37815522427443,
        'V5': -0.338320769942518,
        'V6': 0.462387777762292,
        'V7': 0.239598554061257,
        'V8': 0.0986979012610507,
        'V9': 0.363786969611213,
        'V10': 0.0907941719789316,
        'V11': -0.551599533260813,
        'V12': -0.617800855762348,
        'V13': -0.991389847235408,
        'V14': -0.311169353699879,
        'V15': 1.46817697209427,
        'V16': -0.470400525259478,
        'V17': 0.207971241929242,
        'V18': 0.0257905801985591,
        'V19': 0.403992960255733,
        'V20': 0.251412098239705,
        'V21': -0.018306777944153,
        'V22': 0.277837575558899,
        'V23': -0.110473910188767,
        'V24': 0.0669280749146731,
        'V25': 0.128539358273528,
        'V26': -0.189114843888824,
        'V27': 0.133558376740387,
        'V28': -0.0210530534538215,
        'Amount': 149.62
    }
    
    result = detector.predict(transaction)
    print(f"\nPrediction: {result['label']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Threshold: {result['threshold']:.4f}")
