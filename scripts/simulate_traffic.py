import requests
import time
import random
import numpy as np

API_URL = "http://localhost:8000"

def generate_transaction(drifted=False):
    """Generate a transaction. If drifted, V1 is shifted."""
    # Base legitimate transaction
    tx = {
        "Time": time.time(),
        "Amount": round(random.uniform(10, 500), 2),
    }
    
    # Generate PCA components
    # Normal distribution N(0,1) for standard
    # Shifted N(3,1) for drifted
    mean = 3.0 if drifted else 0.0
    
    for i in range(1, 29):
        key = f"V{i}"
        # We only shift V1 effectively for our drift detector
        val = np.random.normal(mean if i == 1 else 0.0, 1.0)
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

    # 2. Send drifted traffic (50 requests)
    print("Drifted traffic (Shifted V1)...")
    for _ in range(60): # Send enough to influence the buffer significantly
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
