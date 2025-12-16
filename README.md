---
title: Credit Fraud Detection
emoji: 💳
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "3.50.2"
app_file: app/app.py
pinned: false
---

# 💳 Credit Card Fraud Detection - MLOps Project

A fully automated MLOps pipeline for detecting fraudulent credit card transactions using XGBoost, with comprehensive experiment tracking, monitoring, and deployment.

## 🎯 Project Overview

This project implements an end-to-end machine learning operations pipeline for credit card fraud detection, featuring:

- **Model**: XGBoost classifier optimized with Optuna
- **Experiment Tracking**: MLFlow for logging trials, metrics, and artifacts
- **API**: FastAPI for model serving
- **Monitoring**: Prometheus + Grafana for real-time metrics
- **GUI**: Gradio app deployed on Hugging Face Spaces
- **CI/CD**: GitHub Actions for testing, training, and deployment
- **Containerization**: Docker and Docker Compose

## 📊 Dataset

[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle

- **284,807 transactions** (492 frauds, ~0.17% fraud rate)
- **Highly imbalanced** binary classification problem
- **30 features**: Time, Amount, and 28 PCA-transformed features (V1-V28)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional)
- Kaggle API credentials (for dataset download)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd MLOPS-FinalProject
```

2. **Crear entorno virtual con uv**
```bash
# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear entorno e instalar dependencias automáticamente
uv sync

# Activar entorno
source .venv/bin/activate
```

**Nota**: `uv sync` crea automáticamente el entorno virtual e instala todas las dependencias del `pyproject.toml`. Es mucho más rápido que pip.

3. **Download dataset**
```bash
# Set up Kaggle credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

## 📚 Usage

### 1. Data Preprocessing

```bash
python -c "from src.data.preprocessing import DataPreprocessor; DataPreprocessor().preprocess_pipeline()"
```

This will:
- Load and validate the dataset
- Split into train/val/test sets with stratification
- Scale features (Time and Amount)
- Save processed data to `data/processed/`

### 2. Model Training

```bash
python src/models/train.py
```

Features:
- **Stratified K-Fold CV** (5 folds) for robust evaluation
- **Optuna optimization** (50 trials by default)
- **MLFlow tracking** for all experiments
- **Metrics**: ROC-AUC, PR-AUC, F1, Precision, Recall
- **Interpretability**: SHAP values and feature importance
- **Optimal threshold** selection based on F1 score

View experiments:
```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

### 3. Run API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Endpoints:
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make fraud prediction
- `GET /metrics` - Prometheus metrics

Test prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_transaction.json
```

### 4. Run with Docker

```bash
cd docker
docker-compose up -d
```

Services:
- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 5. Monitoring Setup

#### Prometheus
1. Access http://localhost:9090
2. Go to **Status > Targets** to verify API is being scraped
3. Query metrics like `fraud_detection_predictions_total`

#### Grafana
1. Access http://localhost:3000 (login: admin/admin)
2. Add Prometheus data source:
   - URL: `http://prometheus:9090`
3. Create dashboard with panels:
   - **Prediction Volume**: `rate(fraud_detection_predictions_total[5m])`
   - **Fraud Detection Rate**: `fraud_detected_total / (fraud_detected_total + legitimate_transactions_total)`
   - **API Latency**: `histogram_quantile(0.95, prediction_latency_seconds_bucket)`

### 6. Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Specific test file
pytest tests/test_api.py -v
```

## 🏗️ Project Structure

```
.
├── data/
│   ├── raw/              # Raw Kaggle dataset
│   └── processed/        # Processed train/val/test sets
├── src/
│   ├── data/
│   │   └── preprocessing.py    # Data loading and preprocessing
│   ├── models/
│   │   ├── train.py            # Model training with Optuna + MLFlow
│   │   └── inference.py        # Model inference
│   ├── api/
│   │   └── main.py             # FastAPI application
│   └── monitoring/
│       └── metrics.py          # Prometheus metrics
├── tests/
│   ├── test_preprocessing.py
│   ├── test_inference.py
│   └── test_api.py
├── app/
│   ├── app.py                  # Gradio app for Hugging Face
│   └── requirements.txt
├── models/                     # Serialized models
├── mlruns/                     # MLFlow tracking data
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── monitoring/
│   └── prometheus.yml
├── .github/workflows/
│   ├── test.yml               # CI: Run tests
│   ├── train.yml              # Train model
│   └── deploy.yml             # Build and push Docker image
├── requirements.txt
└── README.md
```

## 🔬 Model Details

### Validation Scheme
**Stratified K-Fold Cross-Validation** (5 folds)
- Maintains class distribution across folds
- Critical for imbalanced datasets
- Provides robust performance estimates

### Metrics

**Threshold-Independent** (for hyperparameter optimization):
- **ROC-AUC**: Overall discriminative ability
- **PR-AUC**: Performance on imbalanced data (primary metric)

**Threshold-Dependent** (for production):
- **F1 Score**: Balance of precision and recall
- **Precision**: Minimize false positives
- **Recall**: Catch actual frauds
- **Confusion Matrix**: Detailed error analysis

### Hyperparameters Optimized
- `max_depth`: Tree depth (3-10)
- `learning_rate`: Step size (0.01-0.3)
- `n_estimators`: Number of trees (100-500)
- `min_child_weight`: Minimum sum of instance weight (1-10)
- `subsample`: Row sampling ratio (0.6-1.0)
- `colsample_bytree`: Column sampling ratio (0.6-1.0)
- `gamma`: Minimum loss reduction (0-5)
- `reg_alpha`: L1 regularization (0-2)
- `reg_lambda`: L2 regularization (0-2)
- `scale_pos_weight`: Automatic class imbalance handling

## 📈 Monitoring Metrics

### Counters
- `fraud_detection_predictions_total{prediction_label}`: Total predictions by label
- `fraud_detected_total`: Total frauds detected
- `legitimate_transactions_total`: Total legitimate transactions

### Gauges
- `last_prediction_probability`: Latest fraud probability
- `last_prediction_label`: Latest prediction (0/1)

### Histograms
- `prediction_latency_seconds`: Prediction time distribution

## 🔄 CI/CD Pipeline

### Test Workflow
- **Trigger**: Push/PR to main/develop
- **Steps**: Install deps → Run tests → Upload coverage

### Train Workflow
- **Trigger**: Manual or weekly schedule
- **Steps**: Download data → Preprocess → Train with MLFlow → Upload artifacts
- **Environment**: `MLFLOW_TRACKING_URI` configured for CI

### Deploy Workflow
- **Trigger**: Push to main
- **Steps**: Build Docker image → Push to Docker Hub

## 🎨 Hugging Face Space

Deploy the Gradio app:

1. Create a new Space on Hugging Face
2. Upload `app/app.py` and `app/requirements.txt`
3. Set `API_URL` environment variable to your deployed API
4. Space will auto-deploy

## 🧪 Testing Strategy

- **Unit Tests**: Data preprocessing, inference logic
- **Integration Tests**: API endpoints, request/response validation
- **Mocking**: Model dependencies for fast, isolated tests
- **Coverage**: Aim for >80% code coverage

## 📝 Next Steps

- [ ] Download dataset from Kaggle
- [ ] Train initial model and review MLFlow experiments
- [ ] Deploy API and verify monitoring
- [ ] Upload Gradio app to Hugging Face
- [ ] Set up GitHub repository and configure secrets
- [ ] Run CI/CD pipeline
- [ ] Create Grafana dashboards
- [ ] Write project report

## 🤝 Contributing

This is a final project for MLOps course. For group members:
1. Create feature branches
2. Write tests for new features
3. Submit PRs for review
4. Ensure CI passes before merging

## 📄 License

MIT License

## 🔗 Links

- **GitHub Repository**: [Add your repo URL]
- **Hugging Face Space**: [Add your Space URL]
- **Dataset**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 📧 Contact

[Your Name/Group Members]
[Your Email]

---

**MLOps Final Project** - Fully Automated ML Pipeline for Fraud Detection
