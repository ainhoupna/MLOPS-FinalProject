#!/bin/bash
# Script para generar tráfico y poblar métricas de Prometheus/Grafana

echo "🚀 Generando tráfico para métricas..."
echo ""

# Verificar que la API está corriendo
echo "1️⃣ Verificando API..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: La API no está corriendo en localhost:8000"
    echo "   Ejecuta: cd docker && docker compose up -d"
    exit 1
fi
echo "✅ API está UP"
echo ""

# Generar 200 predicciones (mix de legítimas y fraudes)
echo "2️⃣ Enviando 200 predicciones..."

LEGITIMATE='{"Time":0,"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,"Amount":149.62}'

FRAUD='{"Time":406,"V1":-2.31,"V2":1.95,"V3":-1.61,"V4":3.99,"V5":-0.52,"V6":-1.43,"V7":-2.54,"V8":0.10,"V9":0.44,"V10":-2.42,"V11":-1.03,"V12":0.74,"V13":-1.25,"V14":-2.04,"V15":0.41,"V16":0.14,"V17":0.52,"V18":0.03,"V19":-0.19,"V20":-0.22,"V21":0.50,"V22":0.22,"V23":0.13,"V24":0.25,"V25":0.51,"V26":0.25,"V27":0.04,"V28":0.13,"Amount":0.89}'

for i in {1..200}; do
    # 90% legítimas, 10% fraudes (más realista)
    if [ $((i % 10)) -eq 0 ]; then
        DATA=$FRAUD
    else
        DATA=$LEGITIMATE
    fi
    
    curl -s -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d "$DATA" > /dev/null
    
    # Mostrar progreso cada 20 requests
    if [ $((i % 20)) -eq 0 ]; then
        echo "  📊 $i/200 predicciones enviadas..."
    fi
    
    # Pequeña pausa para simular tráfico real
    sleep 0.05
done

echo ""
echo "✅ 200 predicciones completadas!"
echo ""

echo "3️⃣ Verificando métricas..."
metrics=$(curl -s http://localhost:8000/metrics | grep fraud_detection_predictions_total | head -1)
if [ -n "$metrics" ]; then
    echo "✅ Métricas disponibles en Prometheus"
    echo "   $metrics"
else
    echo "⚠️  Advertencia: No se encontraron métricas"
fi

echo ""
echo "🎉 ¡Listo! Ahora puedes ver las métricas en:"
echo ""
echo "   📊 Prometheus: http://localhost:9091"
echo "      Query: fraud_detection_predictions_total"
echo ""
echo "   📈 Grafana: http://localhost:3000"
echo "      User: admin / Pass: admin"
echo ""
echo "💡 Tip: Espera 30 segundos para que Prometheus scrape las métricas"
