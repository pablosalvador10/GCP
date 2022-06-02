import tempfile
import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import os
import mlflow
from pathlib import Path
import subprocess


def register_dataframe(df, mlflow_artifact_directory="", prefix="", suffix=".csv"):
    temp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix)
    temp_name = temp.name

    try:
        df.to_csv(temp_name, index=False)
        mlflow.log_artifact(temp_name, mlflow_artifact_directory)
    finally:
        temp.close()  # Delete the temp file


def wait_until_ready(client, model_name, model_version):
    model_version_details = client.get_model_version(
        name=model_name,
        version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    # print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status != ModelVersionStatus.READY:
        time.sleep(20)
        self.wait_until_ready(model_name, model_version)


def model_promotion_to_prod(client, model_name, artifact_path):
    ## Check for latest production Model
    latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
    latest_production_version = latest_version_info[0].version
    print("The current production version of the model '%s' is '%s'." % (model_name, latest_production_version))

    # Archive Older Model
    client.transition_model_version_stage(
        name=model_name,
        version=latest_production_version,
        stage="Archived")

    # Productionize newer model
    latest_version_info_staging = client.get_latest_versions(model_name, stages=["Staging"])
    # model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    latest_staging_version = latest_version_info_staging[0].version
    print("The latest staging version of the model '%s' is '%s'." % (model_name, latest_staging_version))

    client.transition_model_version_stage(
        name=model_name,
        version=latest_staging_version,
        stage="Production")

    ### Check for latest production Model
    latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
    latest_production_version = latest_version_info[0].version
    print("The latest production version of the model '%s' is '%s'." % (model_name, latest_production_version))