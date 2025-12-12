#!/bin/bash
# Quick start script

echo "🚀 MLOps Fraud Detection - Quick Start"
echo ""

# Activar entorno
source .venv/bin/activate

# Verificar instalación
echo "✓ Entorno virtual activado"
python --version
echo ""

# Mostrar siguiente pasos
echo "📋 Siguientes pasos:"
echo ""
echo "1. Descargar datos:"
echo "   ./download_data.sh"
echo ""
echo "2. Preprocesar datos:"
echo "   python -c \"from src.data.preprocessing import DataPreprocessor; DataPreprocessor().preprocess_pipeline()\""
echo ""
echo "3. Entrenar modelo:"
echo "   python src/models/train.py"
echo ""
echo "4. Ver experimentos MLFlow:"
echo "   mlflow ui"
echo ""
echo "5. Ejecutar API:"
echo "   uvicorn src.api.main:app --reload"
echo ""
echo "6. Ejecutar tests:"
echo "   pytest tests/ -v"
echo ""
echo "7. Docker (API + Prometheus + Grafana):"
echo "   cd docker && docker-compose up -d"
echo ""
