
import os

import mlflow

from datetime import datetime
import Projects.ml_churn_mlflow.src.utils.mlflow_levi.mlflow_levi as mlflow_l


def registering_job():

    os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:8000"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Pablo"

    client = mlflow.tracking.MlflowClient()

    mlflow.set_experiment("bankchurners_Registering")

    with mlflow.start_run(run_name="Registering") as run:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        user = "Pablo"
        run_id = "96ee1f20ad2b4987910724247d207b63"
        artifact_path = "model"
        model_name = "churncustomers"
        stage = "Staging"
        model_description = f"Model registered from run {run_id} at {date} by {user}"

        # Registering Model
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        mlflow_l.wait_until_ready(client, model_details.name, model_details.version)

        client.transition_model_version_stage(
            name=model_details.name,
            version=model_details.version,
            stage=stage,
        )

        client.update_model_version(
            name=model_details.name,
            version=model_details.version,
            description=model_description
        )
        print(model_details.version)

        # print info run
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        print(run.info)
        mlflow.end_run()