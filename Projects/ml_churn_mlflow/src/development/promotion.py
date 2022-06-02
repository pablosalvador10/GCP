import os
import mlflow
import Projects.ml_churn_mlflow.src.utils.mlflow_levi.mlflow_levi as mlflow_l
from datetime import datetime


def promotion_job():

    os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:8000"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Pablo"

    client = mlflow.tracking.MlflowClient()

    mlflow.set_experiment("bankchurners_promotion")

    with mlflow.start_run(run_name="Promotion") as run:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        user = "Pablo"
        run_id = "96ee1f20ad2b4987910724247d207b63"
        artifact_path = "model"
        model_name = "churncustomers"
        stage = "Staging"
        model_description = f"Model registered from run {run_id} at {date} by {user}"

        mlflow_l.model_promotion_to_prod(client, model_name, artifact_path)

        # print info run
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        print(run.info)
        mlflow.end_run()

