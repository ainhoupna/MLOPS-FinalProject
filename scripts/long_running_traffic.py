import requests
import time
import random
import numpy as np
import sys
from datetime import datetime, timedelta

API_URL = "http://localhost:8000"
DURATION_HOURS = 2

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
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=DURATION_HOURS)
    
    print(f"🚀 Starting traffic simulation at {start_time.strftime('%H:%M:%S')}")
    print(f"⏱️  Will run for {DURATION_HOURS} hours (until {end_time.strftime('%H:%M:%S')})")
    print(f"🎯 Target: {API_URL}/predict")
    print("-" * 50)

    request_count = 0
    errors = 0
    
    try:
        while datetime.now() < end_time:
            # 10% chance of drifted data to make graphs interesting
            is_drifted = random.random() < 0.1
            
            tx = generate_transaction(drifted=is_drifted)
            
            try:
                resp = requests.post(f"{API_URL}/predict", json=tx)
                if resp.status_code == 200:
                    status = "✅"
                else:
                    status = f"❌ ({resp.status_code})"
                    errors += 1
            except Exception as e:
                status = f"❌ ({str(e)})"
                errors += 1
                
            request_count += 1
            
            # Print progress every 10 requests
            if request_count % 10 == 0:
                elapsed = datetime.now() - start_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent {request_count} reqs | Errors: {errors} | Elapsed: {str(elapsed).split('.')[0]}")
            
            # Random sleep between 0.1s and 3s to simulate variable traffic load
            sleep_time = random.uniform(0.1, 3.0)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
    
    print("-" * 50)
    print(f"🏁 Finished. Total requests: {request_count}")

if __name__ == "__main__":
    main()
