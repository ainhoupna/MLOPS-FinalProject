"""
FastAPI application for serving fraud detection model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Dict
import time
import os

from src.models.inference import FraudDetector
from src.monitoring.metrics import metrics_collector


# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0"
)

# Initialize model
detector = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global detector
    try:
        detector = FraudDetector()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        print("API will start but predictions will fail until model is available")


class TransactionFeatures(BaseModel):
    """Input schema for transaction features."""
    Time: float = Field(..., description="Seconds elapsed between this transaction and first transaction")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., description="Transaction amount", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 0,
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061257,
                "V8": 0.0986979012610507,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762348,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525259478,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114843888824,
                "V27": 0.133558376740387,
                "V28": -0.0210530534538215,
                "Amount": 149.62
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction response."""
    prediction: int = Field(..., description="Prediction (0=legitimate, 1=fraud)")
    label: str = Field(..., description="Human-readable label")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    threshold: float = Field(..., description="Classification threshold used")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": float(detector.threshold)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionFeatures):
    """
    Predict if a transaction is fraudulent.
    
    Args:
        transaction: Transaction features
        
    Returns:
        Prediction result with probability and label
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Measure latency
        start_time = time.time()
        
        # Make prediction
        features = transaction.dict()
        result = detector.predict(features)
        
        # Calculate latency
        latency = time.time() - start_time
        latency_ms = latency * 1000
        
        # Record metrics
        metrics_collector.record_prediction(
            prediction=result['prediction'],
            probability=result['fraud_probability'],
            latency=latency
        )
        
        # Add latency to response
        result['latency_ms'] = latency_ms
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Metrics in Prometheus exposition format
    """
    metrics_data = metrics_collector.get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
