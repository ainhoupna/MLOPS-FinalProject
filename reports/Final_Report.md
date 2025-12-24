# 💳 Credit Card Fraud Detection - Final MLOps Report

## 1. Problem Description
The objective of this project is to detect fraudulent credit card transactions. The dataset is highly imbalanced (0.17% fraud), requiring specialized techniques for training and evaluation.

## 2. Model Validation Scheme
- **Methodology**: Stratified K-Fold Cross-Validation (5 folds) to ensure consistent class distribution.
- **Metrics**:
    - **Threshold-Independent**: PR-AUC (Area Under Precision-Recall Curve) was used for hyperparameter optimization as it is more robust than ROC-AUC for imbalanced data.
    - **Threshold-Dependent**: F1-Score, Precision, and Recall were used to select the optimal production threshold.

## 3. Advanced Features (Extras)
To exceed the minimum requirements, we implemented:
- **TabNet**: A deep learning architecture specifically designed for tabular data, providing an alternative to tree-based models.
- **Probability Calibration**: Used `CalibratedClassifierCV` (Sigmoid method) to ensure that the fraud probabilities returned by the model are reliable and well-calibrated.
- **Airflow Orchestration**: A complete DAG (`fraud_detection_pipeline`) that automates data ingestion, preprocessing, and multi-model training.
- **Drift Simulation**: Real-time simulation of Data Drift and Concept Drift, visible in Prometheus and Grafana.

## 4. Testing Logic
- **Unit Tests**: Verified data preprocessing logic and model inference.
- **Integration Tests**: Tested the FastAPI endpoints (`/predict`, `/health`, `/metrics`).
- **CI/CD**: GitHub Actions automatically runs these tests on every push.

## 5. Performance Analysis
- **XGBoost (Calibrated)**: Achieved high PR-AUC (>0.85) with optimized hyperparameters.
- **TabNet**: Provided competitive results, showing the power of deep learning on tabular features.
- **Threshold Optimization**: The final threshold was selected to maximize F1-Score, balancing the cost of false positives vs. missed frauds.

## 6. Monitoring
- **Prometheus**: Scrapes metrics from the API every 10s.
- **Grafana**: Visualizes prediction volume, fraud rates, latency, and **simulated drift**.

> [!IMPORTANT]
> **Links**:
> - **GitHub Repository**: [ainhoupna/MLOPS-FinalProject](https://github.com/ainhoupna/MLOPS-FinalProject)
> - **Hugging Face Space**: [Credit Fraud Detection](https://huggingface.co/spaces/ainhoupna/Credit_Fraud_Detection)

---
*Developed for the MLOps Final Project - 2024/2025*
