"""
Gradio app for Credit Card Fraud Detection - Hugging Face Space
"""

import gradio as gr
import numpy as np
import requests
import os

# API endpoint (can be configured)
API_URL = os.getenv("API_URL", "http://localhost:8000")


def predict_fraud(time, amount, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, 
                  v11, v12, v13, v14, v15, v16, v17, v18, v19, v20):
    """
    Make fraud prediction using the API.
    
    Note: Using only first 20 V features for simpler UI.
    Remaining V21-V28 will be set to 0 for demo purposes.
    """
    # Build feature dictionary
    features = {
        'Time': float(time),
        'Amount': float(amount),
        'V1': float(v1), 'V2': float(v2), 'V3': float(v3), 'V4': float(v4),
        'V5': float(v5), 'V6': float(v6), 'V7': float(v7), 'V8': float(v8),
        'V9': float(v9), 'V10': float(v10), 'V11': float(v11), 'V12': float(v12),
        'V13': float(v13), 'V14': float(v14), 'V15': float(v15), 'V16': float(v16),
        'V17': float(v17), 'V18': float(v18), 'V19': float(v19), 'V20': float(v20),
        # Set remaining features to 0 for demo
        'V21': 0.0, 'V22': 0.0, 'V23': 0.0, 'V24': 0.0,
        'V25': 0.0, 'V26': 0.0, 'V27': 0.0, 'V28': 0.0
    }
    
    try:
        # Call API
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        
        # Format output
        label = result['label']
        probability = result['fraud_probability']
        threshold = result['threshold']
        
        # Color-code based on prediction
        if result['prediction'] == 1:
            status_color = "🔴"
            status_text = f"{status_color} **FRAUD DETECTED**"
        else:
            status_color = "🟢"
            status_text = f"{status_color} **LEGITIMATE TRANSACTION**"
        
        output = f"""
{status_text}

**Fraud Probability:** {probability:.2%}
**Classification Threshold:** {threshold:.4f}
**Latency:** {result.get('latency_ms', 0):.2f} ms
        """
        
        return output
        
    except requests.exceptions.ConnectionError:
        return "❌ **Error:** Cannot connect to API. Please ensure the API is running."
    except requests.exceptions.Timeout:
        return "⏱️ **Error:** Request timeout. The API took too long to respond."
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


# Example transactions
EXAMPLE_LEGITIMATE = [
    0, 149.62,  # Time, Amount
    -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,
    0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47,
    0.21, 0.03, 0.40, 0.25
]

EXAMPLE_FRAUD = [
    406, 0.89,  # Time, Amount
    -2.31, 1.95, -1.61, 3.99, -0.52, -1.43, -2.54, 0.10,
    0.44, -2.42, -1.03, 0.74, -1.25, -2.04, 0.41, 0.14,
    0.52, 0.03, -0.19, -0.22
]


# Create Gradio interface
with gr.Blocks(title="Credit Card Fraud Detection") as demo:
    gr.Markdown("""
    # 💳 Credit Card Fraud Detection
    
    This application uses a machine learning model to detect fraudulent credit card transactions.
    
    **Note:** For simplicity, this demo uses only the first 20 PCA components (V1-V20) and transaction metadata.
    The remaining components (V21-V28) are set to 0.
    
    ### How to use:
    1. Enter transaction details below
    2. Click "Detect Fraud" to get a prediction
    3. Or try the example transactions
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Transaction Details")
            
            time_input = gr.Number(label="Time (seconds since first transaction)", value=0)
            amount_input = gr.Number(label="Amount ($)", value=100.0)
            
            gr.Markdown("### PCA Components (V1-V20)")
            gr.Markdown("*These are anonymized features from PCA transformation*")
            
            with gr.Row():
                v1 = gr.Number(label="V1", value=0.0)
                v2 = gr.Number(label="V2", value=0.0)
                v3 = gr.Number(label="V3", value=0.0)
                v4 = gr.Number(label="V4", value=0.0)
            
            with gr.Row():
                v5 = gr.Number(label="V5", value=0.0)
                v6 = gr.Number(label="V6", value=0.0)
                v7 = gr.Number(label="V7", value=0.0)
                v8 = gr.Number(label="V8", value=0.0)
            
            with gr.Row():
                v9 = gr.Number(label="V9", value=0.0)
                v10 = gr.Number(label="V10", value=0.0)
                v11 = gr.Number(label="V11", value=0.0)
                v12 = gr.Number(label="V12", value=0.0)
            
            with gr.Row():
                v13 = gr.Number(label="V13", value=0.0)
                v14 = gr.Number(label="V14", value=0.0)
                v15 = gr.Number(label="V15", value=0.0)
                v16 = gr.Number(label="V16", value=0.0)
            
            with gr.Row():
                v17 = gr.Number(label="V17", value=0.0)
                v18 = gr.Number(label="V18", value=0.0)
                v19 = gr.Number(label="V19", value=0.0)
                v20 = gr.Number(label="V20", value=0.0)
            
            predict_btn = gr.Button("🔍 Detect Fraud", variant="primary")
        
        with gr.Column():
            gr.Markdown("### Prediction Result")
            output = gr.Markdown()
            
            gr.Markdown("### Try Examples")
            gr.Examples(
                examples=[EXAMPLE_LEGITIMATE, EXAMPLE_FRAUD],
                inputs=[time_input, amount_input, v1, v2, v3, v4, v5, v6, v7, v8,
                       v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20],
                label="Example Transactions"
            )
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_fraud,
        inputs=[time_input, amount_input, v1, v2, v3, v4, v5, v6, v7, v8,
               v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### About the Model
    
    - **Algorithm:** XGBoost Classifier
    - **Optimization:** Optuna for hyperparameter tuning
    - **Tracking:** MLFlow for experiment management
    - **Metrics:** ROC-AUC, PR-AUC, F1-Score
    - **Monitoring:** Prometheus + Grafana
    
    **Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle
    """)


if __name__ == "__main__":
    demo.launch()
