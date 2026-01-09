"""
FastAPI application for serving fraud detection model.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel, Field
import time
import os
import numpy as np
from collections import deque

from src.models.inference import FraudDetector
from src.monitoring.metrics import metrics_collector, drift_detector, fairness_monitor


# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0",
)

# Initialize model and metrics
detector = None

# Monitor top 3 most important features (based on XGBoost feature importance)
MONITORED_FEATURES = ['V14', 'V4', 'V12']

# Separate buffers for each monitored feature
feature_buffers = {
    'V14': deque(maxlen=1000),
    'V4': deque(maxlen=1000),
    'V12': deque(maxlen=1000)
}

# Reference data from training set (simulated with normal distribution N(0,1))
REFERENCE_DATA = {
    'V14': np.random.normal(0, 1, 1000).tolist(),
    'V4': np.random.normal(0, 1, 1000).tolist(),
    'V12': np.random.normal(0, 1, 1000).tolist()
}

# Track prediction distribution for concept drift detection
prediction_buffer = deque(maxlen=1000)  # Store last 1000 predictions
BASELINE_FRAUD_RATE = 0.0017  # Expected fraud rate from training data (0.17%)


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

    Time: float = Field(
        ...,
        description="Seconds elapsed between this transaction and first transaction",
    )
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
                "Amount": 149.62,
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction response."""

    prediction: int = Field(..., description="Prediction (0=legitimate, 1=fraud)")
    label: str = Field(..., description="Human-readable label")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    threshold: float = Field(..., description="Classification threshold used")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve HTML landing page."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback to JSON if HTML not found
        return {
            "message": "Credit Card Fraud Detection API",
            "version": "1.0.0",
            "endpoints": {
                "predict": "/predict",
                "health": "/health",
                "metrics": "/metrics",
                "retrain": "/retrain",
                "docs": "/docs",
            },
            "links": {
                "gradio": "https://huggingface.co/spaces/ainhoupna/Credit_Fraud_Detection",
                "github": "https://github.com/ainhoupna/MLOPS-FinalProject",
            },
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": float(detector.threshold),
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

        # Buffer top 3 features for drift detection
        feature_buffers['V14'].append(transaction.V14)
        feature_buffers['V4'].append(transaction.V4)
        feature_buffers['V12'].append(transaction.V12)
        
        # Buffer prediction for concept drift detection
        prediction_buffer.append(result["prediction"])

        # Calculate latency
        latency = time.time() - start_time
        latency_ms = latency * 1000

        # Record metrics
        metrics_collector.record_prediction(
            prediction=result["prediction"],
            probability=result["fraud_probability"],
            latency=latency,
        )

        # Add latency to response
        result["latency_ms"] = latency_ms

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint with simulated and real drift detection.

    Returns:
        Metrics in Prometheus exposition format
    """
    # 1. Real Data Drift (KS-test on top 3 features: V14, V4, V12)
    drift_scores = []
    for feature in MONITORED_FEATURES:
        if len(feature_buffers[feature]) >= 50:  # Require minimum samples
            score = drift_detector.compute_drift(
                REFERENCE_DATA[feature], 
                list(feature_buffers[feature])
            )
            drift_scores.append(score)
    
    # Average drift across monitored features
    data_drift = np.mean(drift_scores) if drift_scores else 0.0
    
    # 2. Concept Drift (Prediction Distribution Shift)
    # Compare recent fraud prediction rate against baseline
    if len(prediction_buffer) >= 100:  # Need minimum samples for reliable estimate
        recent_fraud_rate = sum(prediction_buffer) / len(prediction_buffer)
        
        # Calculate deviation from baseline (expected 0.17% fraud rate)
        rate_deviation = abs(recent_fraud_rate - BASELINE_FRAUD_RATE)
        
        # Normalize to 0-1 scale (deviation > 0.01 = significant shift)
        # If prediction rate changes by >1%, it indicates concept drift
        concept_drift = min(1.0, rate_deviation / 0.01)
    else:
        concept_drift = 0.0  # Not enough predictions yet
    
    # 3. Simulated Fairness Issue
    fairness_val = fairness_monitor.simulate_fairness_issue()
    
    # Record all
    metrics_collector.record_drift(data_drift, concept_drift)
    metrics_collector.record_fairness(fairness_val)
    
    metrics_data = metrics_collector.get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Trigger model retraining. 
    This is called by Alertmanager when drift/fairness thresholds are breached.
    """
    def mock_retraining_process():
        print("🔄 [RETRAINING] Started automatic retraining process...")
        time.sleep(2)
        print("📥 [RETRAINING] Fetching new data...")
        time.sleep(2)
        print("🧠 [RETRAINING] Training new model version...")
        time.sleep(2)
        print("✅ [RETRAINING] Model successfully retrained and deployed!")
    
    background_tasks.add_task(mock_retraining_process)
    return {"status": "accepted", "message": "Retraining process started"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
