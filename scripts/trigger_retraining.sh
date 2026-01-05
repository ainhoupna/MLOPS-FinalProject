#!/bin/bash
echo "🚨 Simulating Alertmanager webhook call..."
curl -X POST http://localhost:8000/retrain
echo -e "\n✅ Request sent."
