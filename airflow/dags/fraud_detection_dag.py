from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fraud_detection_pipeline',
    default_args=default_args,
    description='End-to-end fraud detection pipeline',
    schedule_interval=timedelta(days=7),
    catchup=False,
)

# 1. Download Data
download_data = BashOperator(
    task_id='download_data',
    bash_command='bash /app/download_data.sh',
    dag=dag,
)

# 2. Preprocess Data
def run_preprocessing():
    from src.data.preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_pipeline()

preprocess_data = PythonOperator(
    task_id='preprocess_data',
    python_callable=run_preprocessing,
    dag=dag,
)

# 3. Train XGBoost (with Calibration)
train_xgboost = BashOperator(
    task_id='train_xgboost',
    bash_command='python /app/src/models/train.py',
    dag=dag,
)

# 4. Train TabNet
train_tabnet = BashOperator(
    task_id='train_tabnet',
    bash_command='python /app/src/models/tabnet_train.py',
    dag=dag,
)

# Define dependencies
download_data >> preprocess_data >> [train_xgboost, train_tabnet]
