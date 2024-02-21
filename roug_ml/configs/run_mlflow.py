"""
Script to run run_mlflow.py
"""

import os

# from roug_ml.configs.my_paths import MLFLOW_BACK_PATH


root_path = os.path.abspath(os.path.join(__file__, os.pardir))
data_path = os.path.abspath(os.path.join(root_path, "../../../data"))
MLFLOW_BACK_PATH = os.path.join(data_path, "artifacts")

# Construct the mlflow server command with the desired artifact path
mlflow_command = f"mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root {MLFLOW_BACK_PATH}/artifacts --host 0.0.0.0 --port 8000"
# mlflow_command = "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://com.diabeloop.dev.datasense.results/flow_models/ --host 0.0.0.0 --port 8000"


# Execute the command
os.system(mlflow_command)
