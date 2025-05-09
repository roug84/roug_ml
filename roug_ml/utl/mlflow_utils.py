from mlflow.tracking import MlflowClient
from beartype.typing import Tuple, List, Optional
import mlflow
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import mlflow

def get_best_run(experiment_name: str, metric_key: str, maximize: bool = True) -> Tuple[str, dict]:
    """
    Retrieves the best run and its parameters from a specified experiment.

    :param experiment_name: Name of the experiment.
    :param metric_key: Key of the metric to use for determining the best run.
    :param maximize: If True, selects the run with the highest metric value; if False, selects the lowest.
    :return: A tuple containing the ID of the best run and the parameters of the best run.
    :raises ValueError: If no such experiment exists or no runs are found.
    """
    client = MlflowClient()

    # Get the experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"No such experiment '{experiment_name}'")

    # Choose sorting direction
    sort_order = "DESC" if maximize else "ASC"

    # Search for runs in the experiment
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metric.{metric_key} {sort_order}"]
    )

    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_params = best_run.data.params

    return best_run_id, best_params



def get_top_n_runs(
    experiment_name: str, metric_key: str, n: int
) -> List[Tuple[str, dict]]:
    """
    Retrieves the top N runs and their parameters from a specified experiment.

    :param experiment_name: Name of the experiment.
    :param metric_key: Key of the metric to use for determining the top runs. The runs are
     ordered by this metric in descending order.
    :param n: Number of top runs to retrieve.
    :return: A list of tuples, each containing the ID of a run and the parameters of that run.
    :raises ValueError: If no such experiment exists or if N is less than or equal to 0.
    """
    if n <= 0:
        raise ValueError("The parameter 'n' should be greater than 0.")

    client = MlflowClient()

    # Get the experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"No such experiment '{experiment_name}'")

    # Search for the best runs in the experiment
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metric.{metric_key} DESC"],
        max_results=n,
    )

    # Extract the run_ids and parameters for the top N runs
    top_n_runs = [(run.info.run_id, run.data.params) for run in runs]

    return top_n_runs


def get_or_create_experiment(name):
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        # The experiment does not exist, create a new one
        experiment_id = mlflow.create_experiment(name)
        print("Experiment created with id: ", experiment_id)
    else:
        # The experiment exists, get its ID
        experiment_id = experiment.experiment_id
        print("Experiment loaded with id: ", experiment_id)

    return experiment_id


def load_top_models(top_n_runs: List[Tuple[str, dict]]) -> List:
    """
    Loads the models for the top N runs using MLflow.

    :param top_n_runs: A list of tuples, each containing the ID of a run and the parameters of that run.
    :return: A list of loaded models.
    """

    loaded_models = []
    for run in top_n_runs:
        run_id, _ = run  # We are only interested in the run ID for loading
        model = mlflow.sklearn.load_model("runs:/{}/pipeline".format(run_id))
        loaded_models.append(model)

    return loaded_models


def register_n_top_models(
        experiment_name: str,
        metric_key: str,
        n: int,
        model_name_prefix: str,
        force: bool = False
) -> List[mlflow.entities.model_registry.ModelVersion]:
    """
    Registers the top N models from an experiment into the MLflow Model Registry.

    :param experiment_name: Name of the experiment
    :param metric_key: Metric used for ranking models
    :param n: Number of top models to register
    :param model_name_prefix: Prefix for registered model names
    :param force: Force
    :return: List of registered model versions
    """
    top_n_runs = get_top_n_runs(experiment_name, metric_key, n)
    registered_models = []

    for i, (run_id, params) in enumerate(top_n_runs, 1):
        model_name = f"{model_name_prefix}_{i}"

        model_version = register_single_model(
            run_id=run_id,
            model_name=model_name,
            metric_key=metric_key,
            ensemble_rank=i,
            experiment_name=experiment_name,
            force=force
        )

        if model_version:
            registered_models.append(model_version)

    return registered_models


def transition_n_models_to_staging(
        model_name_prefix: str,
        n_models: int = 10
) -> None:
    """
    Transitions registered models to staging stage.

    :param model_name_prefix: Prefix used when registering models
    :param n_models: Number of models in ensemble
    """
    for i in range(1, n_models + 1):
        model_name = f"{model_name_prefix}_{i}"
        transition_single_model_to_staging(model_name)


def promote_n_models_to_production(
        model_name_prefix: str,
        n_models: int = 10
) -> None:
    """
    Promotes staging models to production after validation.
    Archives ALL previous versions of the model.

    :param model_name_prefix: Prefix used when registering models
    :param n_models: Number of models in ensemble
    """
    for i in range(1, n_models + 1):
        model_name = f"{model_name_prefix}_{i}"
        promote_single_model_to_production(model_name, archive_existing=True)


def load_production_ensemble_models(
        model_name_prefix: str,
        n_models: int = 10
) -> List:
    """
    Loads all production ensemble models.

    :param model_name_prefix: Prefix used when registering models
    :param n_models: Number of models in ensemble
    :return: List of loaded production models
    """
    models = []

    for i in range(1, n_models + 1):
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name_prefix}_{i}/Production"
        )
        models.append(model)


    return models


def cleanup_model_versions(model_name_prefix: str = "cancer_subtype_predictor", n_models: int = 10):
    """
    Cleanup model versions by keeping only the latest version in Production and archiving others.

    :param model_name_prefix: Prefix used when registering models
    :param n_models: Number of models in ensemble
    """
    client = MlflowClient()

    for i in range(1, n_models + 1):
        model_name = f"{model_name_prefix}_{i}"

        # Get all versions in Production
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])

        if len(prod_versions) > 1:
            # Sort versions by version number (highest first)
            sorted_versions = sorted(prod_versions, key=lambda x: int(x.version), reverse=True)

            # Keep the latest version in Production, archive others
            for version in sorted_versions[1:]:
                print(f"Archiving {model_name} version {version.version}")
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )

            print(f"Kept {model_name} version {sorted_versions[0].version} in Production")


def register_single_model(
        run_id: str,
        model_name: str,
        metric_key: str = None,
        ensemble_rank: int = None,
        experiment_name: str = None,
        force: bool = False
) -> mlflow.entities.model_registry.ModelVersion:
    """
    Registers a single model from a run into the MLflow Model Registry.

    :param run_id: MLflow run ID containing the model
    :param model_name: Name to register the model under
    :param metric_key: Optional metric key used for model evaluation
    :param ensemble_rank: Optional rank in ensemble (if part of ensemble)
    :param experiment_name: Optional name of the experiment
    :param force: If True, register new version even if model exists

    :returns Registered model version
    """
    client = MlflowClient()

    # If force is False, check if model exists and return latest version
    if not force:
        existing_versions = client.get_latest_versions(model_name)
        if existing_versions:
            print(f"Model {model_name} already registered. Use force=True to register new version.")
            return existing_versions[0]

    # Get run details for metadata
    run = client.get_run(run_id)
    metric_value = run.data.metrics.get(metric_key, 0.0) if metric_key else None

    # Prepare tags
    tags = {
        "registration_timestamp": datetime.now().isoformat()
    }
    if metric_key and metric_value is not None:
        tags[metric_key] = str(metric_value)
    if ensemble_rank is not None:
        tags["ensemble_rank"] = str(ensemble_rank)
    if experiment_name:
        tags["experiment_name"] = experiment_name

    # Register the model
    model_version = mlflow.register_model(
        f"runs:/{run_id}/pipeline",
        model_name,
        tags=tags
    )

    return model_version


def transition_single_model_to_staging(
        model_name: str
) -> Optional[mlflow.entities.model_registry.ModelVersion]:
    """
    Transitions a single model to staging stage.

    :param model_name: Name of the registered model

    :returns Transitioned model version if successful, None otherwise
    """
    client = MlflowClient()

    try:
        # Get all versions regardless of stage
        versions = client.get_latest_versions(model_name)
        if not versions:
            print(f"No versions found for model {model_name}")
            return None

        # Sort versions by version number (highest first)
        latest_version = sorted(
            versions,
            key=lambda x: int(x.version),
            reverse=True
        )[0]

        # Only transition if not already in Staging
        if latest_version.current_stage != "Staging":
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Staging"
            )
            print(f"Transitioned {model_name} version {latest_version.version} to Staging")
        else:
            print(f"{model_name} version {latest_version.version} already in Staging")

        return latest_version

    except Exception as e:
        print(f"Error transitioning {model_name} to Staging: {str(e)}")
        return None


def promote_single_model_to_production(
        model_name: str,
        archive_existing: bool = True
) -> Optional[mlflow.entities.model_registry.ModelVersion]:
    """
    Promotes a single staging model to production. Optionally archives all other versions.

    :param model_name: Name of the registered model
    :param archive_existing: If True, archives all other versions of the model

    :returns Promoted model version if successful, None otherwise
    """
    client = MlflowClient()

    try:
        # Get staging versions
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            print(f"No Staging version found for {model_name}")
            return None

        staging_version = staging_versions[0]

        if archive_existing:
            # Get ALL versions of this model
            all_versions = client.search_model_versions(f"name='{model_name}'")

            # Archive all versions except the one we're about to promote
            for version in all_versions:
                if (version.version != staging_version.version and
                        version.current_stage != "Archived"):
                    print(f"Archiving version {version.version} of {model_name}")
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version.version,
                        stage="Archived"
                    )

        # Transition staging version to production
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Production"
        )
        print(f"Promoted {model_name} version {staging_version.version} to Production")

        return staging_version

    except Exception as e:
        print(f"Error promoting {model_name} to Production: {str(e)}")
        return None

# ------------------------------------------- VIZU
# ------------------------------------------- VIZU
# ------------------------------------------- VIZU
# ------------------------------------------- VIZU
# ------------------------------------------- VIZU
# ------------------------------------------- VIZU
# ------------------------------------------- VIZU

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import mlflow


def ensure_numpy_array(y):
    """Convert input to numpy array and ensure it has the right format"""
    if isinstance(y, np.ndarray):
        return y
    return np.array(y)


def create_and_log_roc_curve(y_true, y_score, output_dir, run_id=None):
    """
    Create ROC curve and log it to MLflow

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probability estimates of the positive class)
    output_dir : str
        Directory to save the ROC curve image
    run_id : str, optional
        MLflow run ID to log to

    Returns:
    --------
    float
        ROC AUC score
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure inputs are in the right format
    y_true = ensure_numpy_array(y_true)
    y_score = ensure_numpy_array(y_score)

    # Convert multilabel to binary if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Log to MLflow - don't start a new run, just log to the active run
    try:
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_artifact(roc_path)
    except Exception as e:
        print(f"Warning: Failed to log ROC curve to MLflow: {e}")

    return roc_auc


def create_and_log_pr_curve(y_true, y_score, output_dir, run_id=None):
    """
    Create precision-recall curve and log it to MLflow

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probability estimates of the positive class)
    output_dir : str
        Directory to save the precision-recall curve image
    run_id : str, optional
        MLflow run ID to log to

    Returns:
    --------
    float
        Average precision score
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure inputs are in the right format
    y_true = ensure_numpy_array(y_true)
    y_score = ensure_numpy_array(y_score)

    # Convert multilabel to binary if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # Calculate precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Log to MLflow - don't start a new run, just log to the active run
    try:
        mlflow.log_metric("average_precision", avg_precision)
        mlflow.log_artifact(pr_path)
    except Exception as e:
        print(f"Warning: Failed to log PR curve to MLflow: {e}")

    return avg_precision


def create_and_log_confusion_matrix(y_true, y_pred, output_dir, class_names=None, run_id=None):
    """
    Create confusion matrix and log it to MLflow

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    output_dir : str
        Directory to save the confusion matrix image
    class_names : list, optional
        List of class names for the labels
    run_id : str, optional
        MLflow run ID to log to

    Returns:
    --------
    numpy.ndarray
        Confusion matrix
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure inputs are in the right format
    y_true = ensure_numpy_array(y_true)
    y_pred = ensure_numpy_array(y_pred)

    # Convert multilabel to multiclass if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)

    # Add class labels if provided or auto-generate for small number of classes
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is None and n_classes <= 10:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks + 0.5, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks + 0.5, class_names, rotation=0)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Save the plot
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Log to MLflow - don't start a new run, just log to the active run
    try:
        mlflow.log_artifact(cm_path)
    except Exception as e:
        print(f"Warning: Failed to log confusion matrix to MLflow: {e}")

    return cm


def create_and_log_classification_report(y_true, y_pred, output_dir, run_id=None):
    """
    Create classification report and log it to MLflow

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    output_dir : str
        Directory to save the classification report
    run_id : str, optional
        MLflow run ID to log to

    Returns:
    --------
    dict
        Classification report as a dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure inputs are in the right format
    y_true = ensure_numpy_array(y_true)
    y_pred = ensure_numpy_array(y_pred)

    # Convert multilabel to multiclass if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Generate classification report
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_str = classification_report(y_true, y_pred)

    # Save report as text file
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)

    # Create a more visually appealing report as an image
    plt.figure(figsize=(12, 8))

    # Extract metrics for visualization
    classes = []
    precision = []
    recall = []
    f1_score = []
    support = []

    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            classes.append(class_name)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1_score.append(metrics['f1-score'])
            support.append(metrics['support'])

    # Plot metrics if we have classes to show
    if classes:
        x = np.arange(len(classes))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width, precision, width, label='Precision')
        rects2 = ax.bar(x, recall, width, label='Recall')
        rects3 = ax.bar(x + width, f1_score, width, label='F1-score')

        # Add some text for labels, title and custom x-axis tick labels
        ax.set_ylabel('Scores')
        ax.set_title('Classification Report')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()

        # Add value labels to the bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        fig.tight_layout()

        # Save visualization
        report_viz_path = os.path.join(output_dir, "classification_report_viz.png")
        plt.savefig(report_viz_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Log to MLflow - don't start a new run, just log to the active run
    try:
        mlflow.log_artifact(report_path)
        if classes:  # Only log the visualization if we created it
            mlflow.log_artifact(report_viz_path)

        # Log metrics from classification report
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{class_name}_{metric_name}", value)
    except Exception as e:
        print(f"Warning: Failed to log classification report to MLflow: {e}")

    return report_dict


def create_and_log_training_history(history, output_dir, run_id=None):
    """
    Create training history plots and log them to MLflow

    Parameters:
    -----------
    history : dict
        Dictionary containing training metrics (train_loss, train_acc, val_loss, val_acc)
    output_dir : str
        Directory to save the training history plots
    run_id : str, optional
        MLflow run ID to log to

    Returns:
    --------
    list
        List of paths to the created plots
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    # Check if we have any history to plot
    if not history or not any(len(v) > 0 for v in history.values()):
        return plot_paths

    # Determine how many epochs we have
    max_epochs = max(len(v) for v in history.values())
    epochs = list(range(1, max_epochs + 1))

    # Create loss plot
    if 'train_loss' in history and len(history['train_loss']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs[:len(history['train_loss'])], history['train_loss'], 'b-o', label='Training Loss')

        if 'val_loss' in history and len(history['val_loss']) > 0:
            plt.plot(epochs[:len(history['val_loss'])], history['val_loss'], 'r-s', label='Validation Loss')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        loss_path = os.path.join(output_dir, "loss_history.png")
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(loss_path)

    # Create accuracy plot
    if 'train_acc' in history and len(history['train_acc']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs[:len(history['train_acc'])], history['train_acc'], 'g-o', label='Training Accuracy')

        if 'val_acc' in history and len(history['val_acc']) > 0:
            plt.plot(epochs[:len(history['val_acc'])], history['val_acc'], 'm-s', label='Validation Accuracy')

        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        acc_path = os.path.join(output_dir, "accuracy_history.png")
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(acc_path)

    # Create combined plot
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    if 'train_loss' in history and len(history['train_loss']) > 0:
        plt.plot(epochs[:len(history['train_loss'])], history['train_loss'], 'b-o', label='Training Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(epochs[:len(history['val_loss'])], history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 1, 2)
    if 'train_acc' in history and len(history['train_acc']) > 0:
        plt.plot(epochs[:len(history['train_acc'])], history['train_acc'], 'g-o', label='Training Accuracy')
    if 'val_acc' in history and len(history['val_acc']) > 0:
        plt.plot(epochs[:len(history['val_acc'])], history['val_acc'], 'm-s', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    combined_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths.append(combined_path)

    # Log to MLflow - don't start a new run, just log to the active run
    try:
        # Log artifacts
        for path in plot_paths:
            mlflow.log_artifact(path)

        # Log final epoch metrics
        for metric_name, values in history.items():
            if values:
                mlflow.log_metric(f"final_{metric_name}", values[-1])
    except Exception as e:
        print(f"Warning: Failed to log training history to MLflow: {e}")

    return plot_paths