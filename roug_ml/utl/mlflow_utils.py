from mlflow.tracking import MlflowClient
from beartype.typing import Tuple, List, Optional
import mlflow
from datetime import datetime

import views.views_utl


def get_best_run(experiment_name: str, metric_key: str) -> Tuple[str, dict]:
    """
    Retrieves the best run and its parameters from a specified experiment.

    :param experiment_name: Name of the experiment
    :param metric_key: Key of the metric to use for determining the best run. The best run is
     determined by ordering the runs by this metric in descending order and picking the first one.
    :return: A tuple containing the ID of the best run and the parameters of the best run.
    :raises ValueError: If no such experiment exists.
    """
    client = MlflowClient()

    # Get the experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"No such experiment '{experiment_name}'")

    # Search for the best run in the experiment
    runs = client.search_runs(
        [experiment.experiment_id], order_by=[f"metric.{metric_key} DESC"]
    )

    # Assuming the first run is the best one
    best_run = runs[0]
    # retrieve the best run_id
    best_run_id = best_run.info.run_id
    # retrieve the best parameters
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
