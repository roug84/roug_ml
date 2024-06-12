"""
This file contains main paths
"""

import os

root_path = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to main data folder
data_path = os.path.abspath(os.path.join(root_path, "../../../../data"))

# Path to kaggle folder
KAGGLE_ECG_PATH = os.path.join(data_path, "kaggle")

# Path to results
RESULTS_PATH = os.path.abspath(os.path.join(root_path, "../../../../results"))

# Path to mlflow folder
MLFLOW_BACK_PATH = os.path.join(data_path, "artifacts")

# mlflow.env path
MLFLOW_ENV_PATH = os.path.join(root_path, "mlflow.env")
