import requests
import time
import random
import numpy as np

API_URL = "http://localhost:8000"

def generate_transaction(drifted=False):
    """
    Generate a transaction. 
    If drifted, top 3 features (V14, V4, V12) are shifted.
    """
    # Base transaction
    tx = {
        "Time": time.time(),
        "Amount": round(random.uniform(10, 500), 2),
    }
    
    # Features to shift for drift simulation (top 3 by importance)
    drift_features = [14, 4, 12]
    mean_shift = 3.0 if drifted else 0.0
    
    # Generate all PCA components
    for i in range(1, 29):
        key = f"V{i}"
        # Shift only top 3 features when drifted
        if i in drift_features:
            val = np.random.normal(mean_shift, 1.0)
        else:
            val = np.random.normal(0.0, 1.0)
        tx[key] = float(val)
        
    return tx

def main():
    print(f"🚀 Sending traffic to {API_URL}...")
    
    # 1. Send normal traffic (50 requests)
    print("Normal traffic...")
    for _ in range(50):
        tx = generate_transaction(drifted=False)
        try:
            resp = requests.post(f"{API_URL}/predict", json=tx)
            # print(resp.status_code)
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print("✅ Normal traffic sent.")
    time.sleep(1)
    
    # Check metrics
    try:
        metrics = requests.get(f"{API_URL}/metrics").text
        # Parse drift score simply
        for line in metrics.split("\n"):
            if "data_drift_score" in line and "HELP" not in line and "TYPE" not in line:
                print(f"Current Drift Score (Normal): {line}")
    except:
        print("Could not fetch metrics")

    # 2. Send drifted traffic (60 requests)
    print("Drifted traffic (Shifted V14, V4, V12)...")
    for _ in range(60):  # Send enough to influence the buffer significantly
        tx = generate_transaction(drifted=True)
        try:
            requests.post(f"{API_URL}/predict", json=tx)
        except:
            pass
            
    print("✅ Drifted traffic sent.")
    time.sleep(1)
    
    # Check metrics again
    try:
        metrics = requests.get(f"{API_URL}/metrics").text
        for line in metrics.split("\n"):
            if "data_drift_score" in line and "HELP" not in line and "TYPE" not in line:
                print(f"Current Drift Score (Drifted): {line}")
    except:
        print("Could not fetch metrics")

if __name__ == "__main__":
    main()
