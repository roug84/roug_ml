"""
This script contains the main functions for optimizing ML pipelines
TODO: test functions using unittest.mock or maybe MLFlow serving
"""

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from beartype.typing import List, Dict, Tuple, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from roug_ml.models.pipelines.pytorch_nn_pipeline import NNTorch
from roug_ml.utl.evaluation.eval_utl import calc_loss_acc_val
from roug_ml.utl.mlflow_utils import get_or_create_experiment
from roug_ml.utl.mlflow_utils import (
    create_and_log_roc_curve,
    create_and_log_pr_curve,
    create_and_log_confusion_matrix,
    create_and_log_classification_report,
    create_and_log_training_history
)
import numpy as np
import os
import mlflow

import torch
import random
from roug_ml.utl.set_seed import set_seed
from roug_ml.utl.parameter_utils import flatten_dict
from roug_ml.configs.my_paths import MLFLOW_ENV_PATH
from dotenv import load_dotenv
import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (roc_curve, auc, classification_report,
                            confusion_matrix, precision_recall_curve)
from roug_ml.utl.mlflow_utils import get_or_create_experiment
from roug_ml.utl.parameter_utils import flatten_dict
from roug_ml.configs.my_paths import MLFLOW_ENV_PATH
from dotenv import load_dotenv
import scipy.special
import torch
import torch.nn.functional as F

# Imports for the NNTorch predict_proba method
import torch
import numpy as np
import torch.nn.functional as F

# Imports for training visualization utilities
import os
import matplotlib.pyplot as plt
import numpy as np
import mlflow

# Imports for model diagnostic utilities
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve,
                           auc, average_precision_score, f1_score)
import mlflow

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(123)
random.seed(123)

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

set_seed(42)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import numpy as np


def calc_loss_acc_val_dynamic(predictions, targets, metric="accuracy"):
    if metric == "accuracy":
        return accuracy_score(targets, predictions)
    elif metric == "rmse":
        return np.sqrt(mean_squared_error(targets, predictions))
    elif metric == "mae":
        return mean_absolute_error(targets, predictions)
    elif metric == "r2":
        return r2_score(targets, predictions)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def parallele_hyper_optim(
    task_type: str,
    in_num_workers: int,
    x_train: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    param_grid_outer: List[Dict],
    in_framework: str,
    model_save_path: str,
    in_mlflow_experiment_id: str,
    in_mlflow_experiment_name: str,
    use_kfold: bool = False,
    dataset=None,
    data_collator=None,
    in_scaler: Optional[Union[BaseEstimator, TransformerMixin]] = None,
    in_selector: Optional[Union[BaseEstimator, TransformerMixin]] = None,
) -> List[Tuple[dict, float]]:
    """
    Perform parallel hyperparameter optimization for a given framework.

    :param in_num_workers: Number of parallel workers.
    :param x_train: Training data.
    :param y: Training labels.
    :param x_val: Validation data.
    :param y_val: Validation labels.
    :param param_grid_outer: Outer parameter grid. This is the hyperparameter space over which
                             the optimization process will search.
    :param in_framework: Framework to use for optimization ('tf' or 'torch').
    :param model_save_path: Path where the trained models will be saved.
    :param in_mlflow_experiment_id: id of the MLFlow experiment to log the optimization
                                      results.
    :param in_mlflow_experiment_name: Name of the MLFlow experiment to log the optimization
                                      results.
    :param use_kfold: If True, apply K-Fold Cross Validation during the optimization process.
                      Default is False.
    :param in_scaler: Scaler object to be used for feature scaling.
    :param in_selector: Feature selector to be used for feature selection.

    :return: List of tuples containing optimized parameters and corresponding validation accuracies.
    """
    if use_kfold:
        return Parallel(n_jobs=in_num_workers)(
            delayed(kfold_training_with_process_params_torch)(
                params=params_outer,
                x_train=x_train,
                y=y,
                train_loader=params_outer,
                k_folds=5,
                dataset=dataset,
                model_save_path=model_save_path,
                in_mlflow_experiment_id=in_mlflow_experiment_id,
                in_mlflow_experiment_name=in_mlflow_experiment_name,
                in_scaler=in_scaler,
                in_selector=in_selector,
            )
            for params_outer in param_grid_outer
        )
    else:
        # Perform parallel processing for Torch framework
        # return Parallel(n_jobs=in_num_workers)(
        #     delayed(process_params_torch)(params_outer, x_train, y, x_val, y_val, model_save_path,
        #                                   in_mlflow_experiment_id, use_kfold, dataset,
        #                                    None, in_scaler, in_selector
        #                                   )
        #     for params_outer in param_grid_outer
        # )
        results = []
        for params_outer in param_grid_outer:
            result = process_params_torch(task_type,
                params_outer, x_train, y, x_val, y_val, model_save_path,
                in_mlflow_experiment_id, in_mlflow_experiment_name, use_kfold, dataset, data_collator,
                None, in_scaler, in_selector
            )
            results.append(result)
        # result = process_params_torch(param_grid_outer, x_train, y, x_val, y_val,
        #                               model_save_path, in_mlflow_experiment_id,
        #                               in_scaler=in_scaler, in_selector=in_selector)
        # results.append(result)
        return results


def create_and_fit_pipeline(
    task_type: str,
    params: Dict[str, Union[int, str, list]],
    x_train: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_save_path,
    dataset=None,
    data_collator=None,
    in_scaler: Optional[Union[BaseEstimator, TransformerMixin]] = None,
    in_selector: Optional[Union[BaseEstimator, TransformerMixin]] = None,
) -> Tuple:
    """
    Create a pipeline and fit it to the training data.
    :param params: dict, Parameters for the neural network. This should include
        'activations', 'in_nn', 'input_shape', 'output_shape',
        'batch_size', 'cost_function', 'learning_rate',
        'metrics', 'n_epochs', 'nn_key'.
    :param x_train: np.ndarray, The training data.
    :param y: np.ndarray, The training labels.
    :param x_val: np.ndarray, The validation data.
    :param y_val: np.ndarray, The validation labels.
    :param in_scaler: Scaler object to be used for feature scaling.
    :param in_selector: Feature selector to be used for feature selection.

    :return: The pipeline created and fitted, and the validation accuracy of the fitted pipeline.
    """
    x_val_transformed = x_val
    steps = []
    # Check if a scaler is provided
    if in_scaler is not None:
        in_scaler.fit(x_train)
        x_train = in_scaler.transforms(x_train)
        steps.append(("scaler", in_scaler))

    # Check if a feature selector is provided
    if in_selector is not None:
        # Fit the feature selector
        in_selector.fit(x_train, y)

        # Get the shape of the transformed data
        x_train = in_selector.transform(x_train)
        new_input_shape = x_train.shape[1]

        # Transforms validation data
        x_val_transformed = in_selector.transform(x_val)

        # Adjust the input_shape NN based
        if isinstance(params, list):
            for i in range(len(params)):
                params[i]["nn_params"]["input_shape"] = new_input_shape
        else:
            params["nn_params"]["input_shape"] = new_input_shape
        steps.append(("feature_selection", in_selector))

    # Check if dataset and model path are provided in params
    params["task_type"] = task_type

    if dataset is not None and model_save_path is not None:
        torch_stimator = NNTorch(**params)
        torch_stimator.fit(X=None, y=None, **{"dataset": dataset, "data_collator": data_collator, "model_path": model_save_path})
    else:
        # add estimator
        torch_stimator = NNTorch(**params)
        torch_stimator.fit(x_train, y, **{"validation_data": (x_val_transformed, y_val)})

    steps.append(("estimator", torch_stimator))

    pipeline_torch = Pipeline(steps=steps)
    # pipeline_torch.predict(x_val)
    # if x_val is not None and y_val is not None:
    #     predictions = pipeline_torch.predict(x_val)
    #     val_acc = calc_loss_acc_val(predictions, y_val)
    if x_val is not None and y_val is not None:
        predictions = pipeline_torch.predict(x_val_transformed)
        metric = params.get("metrics", "accuracy")
        val_score = calc_loss_acc_val_dynamic(predictions, y_val, metric=metric)
    else:
        val_score = -1
    return pipeline_torch, val_score


def process_params_torch(
        task_type: str,
        params: Dict[str, Union[int, str, list]],
        x_train: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        model_save_path: str,
        in_mlflow_experiment_id: str = "name",
        in_mlflow_experiment_name: str = "name",
        is_fold: bool = False,
        dataset=None,
        data_collator=None,
        val_all_folds: float or None = None,
        in_scaler: Optional[Union[BaseEstimator, TransformerMixin]] = None,
        in_selector: Optional[Union[BaseEstimator, TransformerMixin]] = None,
) -> Tuple[
    Dict[str, Union[int, str, list]] or None,
    float,
    str or None,
]:
    """
    Process parameters and compute validation accuracy for Torch pipeline.

    :param params: Parameters for the neural network. This should include
        'activations', 'in_nn', 'input_shape', 'output_shape',
        'batch_size', 'cost_function', 'learning_rate',
        'metrics', 'n_epochs', 'nn_key'.
    :param x_train:The training data.
    :param y: The training labels.
    :param x_val: The validation data.
    :param y_val: The validation labels.
    :param model_save_path: str, The path where the trained model should be saved.
    :param in_mlflow_experiment_name: str, optional, The name of the MLflow experiment where results
        are logged. Default is 'my_experiment_2'.
    :param is_fold: bool, optional, Flag to indicate whether the process is part of a
        cross-validation fold. Default is False.
    :param val_all_folds: float or None, optional, If not None, this accuracy value will override
        the validation accuracy calculated within the function. Default is None.
    :param in_scaler: Scaler object to be used for feature scaling.
    :param in_selector: Feature selector to be used for feature selection.

    :return: A tuple of the processed parameters, validation accuracy, and run ID.
    """
    from roug_ml.utl.mlflow_utils import (
        create_and_log_roc_curve,
        create_and_log_pr_curve,
        create_and_log_confusion_matrix,
        create_and_log_classification_report,
        create_and_log_training_history
    )

    params = params.copy()

    # Ensure the model save path exists
    os.makedirs(model_save_path, exist_ok=True)

    # Create and train pipeline
    pipeline_torch, val_accuracy = create_and_fit_pipeline(task_type,
        params, x_train, y, x_val, y_val, model_save_path, dataset, data_collator, in_scaler, in_selector
    )

    # If this is a fold from cross-validation, just return without MLflow logging
    if is_fold:
        return None, val_accuracy, None
    else:
        # If it's not a fold but we have k-fold validation, use the mean accuracy
        if val_all_folds is not None:
            val_accuracy = val_all_folds

    # Get model predictions
    y_pred = pipeline_torch.predict(x_val)

    # Try to get probability predictions for ROC curve
    try:
        # Get the estimator (NNTorch) from the pipeline
        nn_model = pipeline_torch.named_steps['estimator']
        # Use predict_proba method to get class probabilities
        y_pred_proba = nn_model.predict_proba(x_val)
        # For binary classification, we need the positive class probabilities
        if y_pred_proba.shape[1] == 2:
            # Take the positive class probability (second column)
            pos_probs = y_pred_proba[:, 1]
        else:
            # If we have a single column, assume these are already the positive probabilities
            pos_probs = y_pred_proba
        has_probas = True
    except (AttributeError, IndexError, KeyError, Exception) as e:
        print(f"Warning: Could not compute probability predictions: {e}")
        # If we can't get probabilities, use the binary predictions
        pos_probs = y_pred
        has_probas = False

    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://localhost:8000")
    load_dotenv(MLFLOW_ENV_PATH)
    mlflow_experiment_id = get_or_create_experiment(in_mlflow_experiment_name)

    # Start MLflow run
    with mlflow.start_run(experiment_id=mlflow_experiment_id) as run:
        print("Run ID:", run.info.run_id)

        # Log parameters
        mlflow.log_params(flatten_dict(params))

        # Log validation accuracy
        mlflow.log_metric("val_accuracy", val_accuracy)

        # Log the trained model
        mlflow.sklearn.log_model(pipeline_torch, "pipeline")

        # Create and log visualizations - all within the same MLflow run
        # The visualization functions now handle their own errors and don't start new runs

        # If we have a binary classification problem with probability estimates
        if len(np.unique(y_val)) == 2 and has_probas:
            create_and_log_roc_curve(y_val, pos_probs, model_save_path)
            create_and_log_pr_curve(y_val, pos_probs, model_save_path)

        # Confusion matrix and classification report for any classification task
        # create_and_log_confusion_matrix(y_val, y_pred, model_save_path)
        # create_and_log_classification_report(y_val, y_pred, model_save_path)

        # Extract training history from model if available
        try:
            nn_model = pipeline_torch.named_steps['estimator']

            # Try to get history from the model
            if hasattr(nn_model, 'history') and nn_model.history:
                create_and_log_training_history(nn_model.history, model_save_path)
            # Fall back to val_accuracies if history isn't available
            elif hasattr(nn_model, 'val_accuracies') and nn_model.val_accuracies:
                # Create basic history with just validation accuracy
                basic_history = {
                    'val_acc': nn_model.val_accuracies
                }
                create_and_log_training_history(basic_history, model_save_path)

                # Log the final validation accuracy
                mlflow.log_metric("final_val_accuracy", nn_model.val_accuracies[-1])
        except Exception as e:
            print(f"Warning: Could not log training history: {e}")

    return params, val_accuracy, run.info.run_id
# def process_params_torch(
#     params: Dict[str, Union[int, str, list]],
#     x_train: np.ndarray,
#     y: np.ndarray,
#     x_val: np.ndarray,
#     y_val: np.ndarray,
#     model_save_path: str,
#     in_mlflow_experiment_id: str = "name",
#     in_mlflow_experiment_name: str = "name",
#     is_fold: bool = False,
#     dataset = None,
#     data_collator = None,
#     val_all_folds: float or None = None,
#     in_scaler: Optional[Union[BaseEstimator, TransformerMixin]] = None,
#     in_selector: Optional[Union[BaseEstimator, TransformerMixin]] = None,
# ) -> Tuple[
#     Dict[str, Union[int, str, list]] or None,
#     float,
#     str or None,
# ]:
#     """
#     Process parameters and compute validation accuracy for Torch pipeline.
#
#     :param params: Parameters for the neural network. This should include
#         'activations', 'in_nn', 'input_shape', 'output_shape',
#         'batch_size', 'cost_function', 'learning_rate',
#         'metrics', 'n_epochs', 'nn_key'.
#     :param x_train:The training data.
#     :param y: The training labels.
#     :param x_val: The validation data.
#     :param y_val: The validation labels.
#     :param model_save_path: str, The path where the trained model should be saved.
#     :param in_mlflow_experiment_name: str, optional, The name of the MLflow experiment where results
#         are logged. Default is 'my_experiment_2'.
#     :param is_fold: bool, optional, Flag to indicate whether the process is part of a
#         cross-validation fold. Default is False.
#     :param val_all_folds: float or None, optional, If not None, this accuracy value will override
#         the validation accuracy calculated within the function. Default is None.
#     :param in_scaler: Scaler object to be used for feature scaling.
#     :param in_selector: Feature selector to be used for feature selection.
#
#     :return: A tuple of the processed parameters, validation accuracy, and run ID.
#     """
#     params = params.copy()
#
#     pipeline_torch, val_accuracy = create_and_fit_pipeline(
#         params, x_train, y, x_val, y_val, model_save_path, dataset, data_collator, in_scaler, in_selector
#     )
#     # If it is a fold from fold validation then do not save the model to MLflow
#     if is_fold:
#         return None, val_accuracy, None
#     else:
#         # if it is not a fold, but it is still k-fold validation then the validation is replaced by
#         # the mean of all folds
#         if val_all_folds is not None:
#             val_accuracy = val_all_folds
#
#     # model_name = 'model_' + '_'.join([f"{k}_{v}" for k, v in params.items()])
#     # pipeline_torch.named_steps['NN'].save(model_save_path, model_name)
#
#     mlflow_experiment_name = in_mlflow_experiment_name
#
#     mlflow.set_tracking_uri("http://localhost:8000")
#     # Ensure environment variables are set for MinIO
#     load_dotenv(MLFLOW_ENV_PATH)
#     # os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
#     # os.environ["AWS_ACCESS_KEY_ID"] = "mlflow_user"
#     # os.environ["AWS_SECRET_ACCESS_KEY"] = "H03042020P16082022"
#
#     # mlflow.set_tracking_uri("http://public_ip_where_ml_flow_is_running:8000") <- public ip of VM
#     # do not forget to do: export PATH=$PATH:/home/ubuntu/.local/bin in VM
#
#     # mlflow.set_tracking_uri('http://host.docker.internal:8000')
#     mlflow_experiment_id = get_or_create_experiment(mlflow_experiment_name)
#     with mlflow.start_run(experiment_id=mlflow_experiment_id) as run:
#         print("Run ID:", run.info.run_id)
#         mlflow.log_params(flatten_dict(params))
#         mlflow.log_metric("val_accuracy", val_accuracy)
#         mlflow.sklearn.log_model(pipeline_torch, "pipeline")
#         # mlflow.pytorch.log_model(pipeline_torch.named_steps['NN'].nn_model, "models")
#         for k, v in params.items():
#             mlflow.log_param(k, v)
#
#     return params, val_accuracy, run.info.run_id
#

def kfold_training_with_process_params_torch(
    params: Dict[str, Union[int, str, list]],
    x_train: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    train_loader: Optional[DataLoader] = None,
    k_folds: int = 5,
    model_save_path: str = "models",
    in_mlflow_experiment_id: str = "name",
    in_mlflow_experiment_name: str = "name",
    in_scaler: Optional[Union[BaseEstimator, TransformerMixin]] = None,
    in_selector: Optional[Union[BaseEstimator, TransformerMixin]] = None,
) -> None:
    """
    Perform k-fold cross-validation training using the `process_params_torch` function.

    :param params: dict, Parameters for the neural network.
    :param x_train: np.ndarray, Training data.
    :param y: np.ndarray, Training labels.
    :param train_loader: Dataset for training.
    :param k_folds: Number of folds to use for the cross-validation, defaults to 5
    :param model_save_path: str, The path where the trained model should be saved.
    :param in_mlflow_experiment_id: id of the MLFlow experiment to log the optimization
                                      results.
    :param in_mlflow_experiment_name: Name of the MLFlow experiment to log the optimization
                                      results.
    :param in_scaler: Scaler object to be used for feature scaling.
    :param in_selector: Feature selector to be used for feature selection.

    """
    # Assert X and y are numpy arrays.
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y, np.ndarray)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    vals = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(x_train)):
        # Sample elements randomly from a given list of ids, no replacement.
        x_train_fold = x_train[train_ids]
        y_train_fold = y[train_ids]

        x_val_fold = x_train[val_ids]
        y_val_fold = y[val_ids]

        # Use the process_params_torch function for training and validation
        processed_params, val_accuracy, run_id = process_params_torch(
            params,
            x_train_fold,
            y_train_fold,
            x_val_fold,
            y_val_fold,
            model_save_path,
            in_mlflow_experiment_id,
            in_mlflow_experiment_name,
            is_fold=True,
            val_all_folds=None,
            in_scaler=in_scaler,
            in_selector=in_selector,
        )
        print("--------------------------------")
        print(f"Validation Accuracy on FOLD {fold}: {val_accuracy}")
        vals.append(val_accuracy)
        print("--------------------------------")
    mean_val = np.mean(vals)

    # Use the process_params_torch function for training and validation
    processed_params, val_accuracy, run_id = process_params_torch(
        params,
        x_train,
        y,
        x_train,
        y,
        model_save_path,
        in_mlflow_experiment_name,
        is_fold=False,
        val_all_folds=mean_val,
        in_scaler=in_scaler,
        in_selector=in_selector,
    )
    print("--------------------------------")
    print(f"Mean validation Accuracy: {mean_val}")
    print("--------------------------------")

    return params, val_accuracy, run_id


def get_best_run_from_hyperoptim(
    results: List[Tuple[Dict, float, str]],
    maximize: bool = True
) -> Tuple[Dict, float, str]:
    """
    This function retrieves the best model run from the results of the hyperparameter optimization.

    :param results: parameters, validation metric, and run ID for each model run
    :param maximize: If True, selects the highest metric (e.g. accuracy); if False, selects the lowest (e.g. loss)

    :return: best_params: the best parameters from hyperparameter optimization
             best_val_score: the best metric value
             best_run_id: ID of the best run
    """
    best_params = {}
    best_val_score = float("-inf") if maximize else float("inf")
    best_run_id = None

    for params, val_score, run_id in results:
        if (maximize and val_score > best_val_score) or (not maximize and val_score < best_val_score):
            best_val_score = val_score
            best_params = params
            best_run_id = run_id

    return best_params, best_val_score, best_run_id

