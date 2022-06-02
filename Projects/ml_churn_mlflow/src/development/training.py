import os
import pandas as pd
import mlflow
import tempfile
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
import Projects.ml_churn_mlflow.src.utils.helpers.model_evaluation as m_eval
import Projects.ml_churn_mlflow.src.utils.mlflow_levi.mlflow_levi as mlflow_l
from sklearn.model_selection import RandomizedSearchCV



def training_job(location):

    os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:8000"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Pablo"

    client = mlflow.tracking.MlflowClient()

    mlflow.set_experiment("bankchurners_training")

    with mlflow.start_run(run_name="Training") as run:
        # consume features

        df = pd.read_csv(location, na_values=["n.a.", "?", "NA", "n/a", "na", "--", "nan"], index_col=False)

        # Train/test Split

        X = df.drop("Attrition_Flag", axis=1)
        y = df.pop("Attrition_Flag")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1, stratify=y)

        # register Train/Test



        mlflow_l.register_dataframe(X_train, mlflow_artifact_directory="Training_data", prefix="Training_Data_Features_",
                             suffix=".csv")
        mlflow_l.register_dataframe(y_train, mlflow_artifact_directory="Training_data", prefix="Training_Data_Target_",
                             suffix=".csv")
        mlflow_l.register_dataframe(X_test, mlflow_artifact_directory="Training_data", prefix="Test_Data_Features_", suffix=".csv")
        mlflow_l.register_dataframe(y_test, mlflow_artifact_directory="Training_data", prefix="Test_Data_Target_", suffix=".csv")

        # Choose the type of classifier.
        bagging_estimator_tuned = BaggingClassifier(random_state=1, n_jobs=-1)

        # Grid of parameters to choose from
        parameters = {'max_samples': [0.1, 0.6, 0.9, 1],
                      'max_features': [0.1, 0.6, 0.8, 0.9, 1],
                      'n_estimators': [10, 20, 40, 50, 100],
                      }

        # Type of scoring used to compare parameter combinations
        acc_scorer = metrics.make_scorer(metrics.recall_score)
        n_iter_search = 50
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
        # Run the grid search
        grid_obj = RandomizedSearchCV(bagging_estimator_tuned, n_iter=n_iter_search, param_distributions=parameters,
                                      verbose=0, scoring=acc_scorer, cv=5, n_jobs=4)
        grid_obj = grid_obj.fit(X_train, y_train)

        # Set the clf to the best combination of parameters
        bagging_estimator_tuned = grid_obj.best_estimator_

        # Fit the best algorithm to the data.
        bagging_estimator_tuned.fit(X_train, y_train)

        # Logging Model
        mlflow.sklearn.log_model(bagging_estimator_tuned, "model")

        ## Evaluation

        test_acc, test_recall, test_precision, bagging_estimator_score = m_eval.get_metrics_score(bagging_estimator_tuned,
                                                                                                  X_train, X_test, y_train,
                                                                                                  y_test)
        cm = m_eval.make_confusion_matrix(bagging_estimator_tuned, X_test, y_test)

        temp = tempfile.NamedTemporaryFile(prefix="cm", suffix=".jpeg")
        temp_name = temp.name

        try:
            cm.figure.savefig(temp_name)
            mlflow.log_artifact(temp_name, "Evaluation")
        finally:
            temp.close()  # Delete the temp file

        # Logging Metrics
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_precision", test_precision)

        # Logging Parameters
        params = bagging_estimator_tuned.get_params()
        for i in params:
            mlflow.log_param(i, params[i])

        # print info run
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        print(run.info)
        mlflow.end_run()