"""
Gradio app for Credit Card Fraud Detection - Hugging Face Space
"""

import gradio as gr
import requests
import os

# API endpoint (can be configured)
API_URL = os.getenv("API_URL", "https://mlops-final-project-latest.onrender.com")


def predict_fraud(
    time,
    amount,
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
    v21, v22, v23, v24, v25, v26, v27, v28,
):
    """
    Make fraud prediction using the API with all 30 features.
    """
    # Build feature dictionary with all features
    features = {
        "Time": float(time),
        "Amount": float(amount),
        "V1": float(v1),
        "V2": float(v2),
        "V3": float(v3),
        "V4": float(v4),
        "V5": float(v5),
        "V6": float(v6),
        "V7": float(v7),
        "V8": float(v8),
        "V9": float(v9),
        "V10": float(v10),
        "V11": float(v11),
        "V12": float(v12),
        "V13": float(v13),
        "V14": float(v14),
        "V15": float(v15),
        "V16": float(v16),
        "V17": float(v17),
        "V18": float(v18),
        "V19": float(v19),
        "V20": float(v20),
        "V21": float(v21),
        "V22": float(v22),
        "V23": float(v23),
        "V24": float(v24),
        "V25": float(v25),
        "V26": float(v26),
        "V27": float(v27),
        "V28": float(v28),
    }

    try:
        # Call API
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        response.raise_for_status()

        result = response.json()

        # Format output
        # label = result["label"]
        probability = result["fraud_probability"]
        threshold = result["threshold"]

        # Color-code based on prediction
        is_fraud = result.get("is_fraud", False)
        if is_fraud:
            status_color = "🔴"
            status_text = f"{status_color} **FRAUD DETECTED**"
        else:
            status_color = "🟢"
            status_text = f"{status_color} **LEGITIMATE TRANSACTION**"

        output = f"""
{status_text}

**Fraud Probability:** {probability:.2%}
**Classification Threshold:** {threshold:.4f}
**Latency:** {result.get("latency_ms", 0):.2f} ms
        """

        return output

    except requests.exceptions.ConnectionError:
        return "❌ **Error:** Cannot connect to API. Please ensure the API is running."
    except requests.exceptions.Timeout:
        return "⏱️ **Error:** Request timeout. The API took too long to respond."
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


# Example transactions with all 30 features
EXAMPLE_LEGITIMATE = [
    0,  # Time
    149.62,  # Amount
    -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09,  # V1-V10
    -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25,  # V11-V20
    -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, -0.02,  # V21-V28
]

EXAMPLE_FRAUD = [
    406,  # Time
    0.89,  # Amount
    -2.31, 1.95, -1.61, 3.99, -0.52, -1.43, -2.54, 0.10, 0.44, -2.42,  # V1-V10
    -1.03, 0.74, -1.25, -2.04, 0.41, 0.14, 0.52, 0.03, -0.19, -0.22,  # V11-V20
    0.50, 0.22, 0.13, 0.25, 0.51, 0.25, 0.04, 0.13,  # V21-V28
]


# Create Gradio interface
with gr.Blocks(title="Credit Card Fraud Detection") as demo:
    gr.Markdown(
        """
    # 💳 Credit Card Fraud Detection

    This application uses a machine learning model to detect fraudulent credit card transactions.
    Enter all 28 PCA components (V1-V28) along with Time and Amount for accurate predictions.

    ### How to use:
    1. Enter transaction details below
    2. Click "Detect Fraud" to get a prediction
    3. Or try the example transactions
    """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Transaction Details")

            time_input = gr.Number(
                label="Time (seconds since first transaction)", value=0
            )
            amount_input = gr.Number(label="Amount ($)", value=100.0)

            gr.Markdown("### PCA Components (V1-V28)")
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
            
            with gr.Row():
                v21 = gr.Number(label="V21", value=0.0)
                v22 = gr.Number(label="V22", value=0.0)
                v23 = gr.Number(label="V23", value=0.0)
                v24 = gr.Number(label="V24", value=0.0)
            
            with gr.Row():
                v25 = gr.Number(label="V25", value=0.0)
                v26 = gr.Number(label="V26", value=0.0)
                v27 = gr.Number(label="V27", value=0.0)
                v28 = gr.Number(label="V28", value=0.0)

            predict_btn = gr.Button("🔍 Detect Fraud", variant="primary")

        with gr.Column():
            gr.Markdown("### Prediction Result")
            output = gr.Markdown()

            gr.Markdown("### Try Examples")
            gr.Examples(
                examples=[EXAMPLE_LEGITIMATE, EXAMPLE_FRAUD],
                inputs=[
                    time_input, amount_input,
                    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                    v21, v22, v23, v24, v25, v26, v27, v28,
                ],
                label="Example Transactions",
            )

    # Connect button to prediction function
    predict_btn.click(
        fn=predict_fraud,
        inputs=[
            time_input, amount_input,
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28,
        ],
        outputs=output,
    )

    gr.Markdown(
        """
    ---
    ### About the Model
    - **Algorithm:** XGBoost Classifier
    - **Optimization:** Optuna for hyperparameter tuning
    - **Tracking:** MLFlow for experiment management
    - **Metrics:** ROC-AUC, PR-AUC, F1-Score
    - **Monitoring:** Prometheus + Grafana

    **Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle
    """
    )


if __name__ == "__main__":
    demo.launch()
