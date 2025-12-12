#!/bin/bash
# Script para descargar el dataset de Kaggle

echo "📥 Descargando Credit Card Fraud Detection dataset..."

# Verificar que kaggle está configurado
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Error: No se encuentra ~/.kaggle/kaggle.json"
    echo ""
    echo "Por favor, configura tus credenciales de Kaggle:"
    echo "1. Ve a https://www.kaggle.com/settings/account"
    echo "2. Crea un nuevo API token (descargará kaggle.json)"
    echo "3. Ejecuta:"
    echo "   mkdir -p ~/.kaggle"
    echo "   mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Crear directorio si no existe
mkdir -p data/raw

# Descargar dataset
source .venv/bin/activate
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

echo "✅ Dataset descargado en data/raw/creditcard.csv"
