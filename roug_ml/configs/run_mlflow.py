"""
Script to run MLflow Tracking Server
"""

import os
from dotenv import load_dotenv

load_dotenv("/Users/hector/cancer_subtype_prediction/roug_ml/roug_ml/configs/mlflow.env")
# Construct the mlflow server command with the desired artifact path and PostgreSQL backend store
mlflow_command = (
    "mlflow server --backend-store-uri postgresql://mlflow_user:H03042020P16082022@localhost/mlflow_db "
    "--default-artifact-root s3://mlflow/artifacts --host 0.0.0.0 --port 8000"
)

# Execute the command
os.system(mlflow_command)
