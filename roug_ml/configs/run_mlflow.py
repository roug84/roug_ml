"""
Script to run MLflow Tracking Server
"""

import os
from dotenv import load_dotenv

load_dotenv("/home/hector/roug/cancer_subtype_prediction/roug_ml/roug_ml/configs/mlflow.env")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "kensou"
os.environ["AWS_SECRET_ACCESS_KEY"] = "H03042020P16082022"

# Construct the mlflow server command with the desired artifact path and PostgreSQL backend store
mlflow_command = (
    "mlflow server --backend-store-uri postgresql://mlflow_user:H03042020P16082022@localhost/mlflow_db "
    "--default-artifact-root s3://mlflow/artifacts --host 0.0.0.0 --port 8000"
)

# Execute the command
os.system(mlflow_command)
