import os
import pandas as pd
import mlflow
import tempfile
from sklearn import preprocessing


def Feature_engineering(location):
    # load the data
    df = pd.read_csv(f"{location}", na_values=["n.a.", "?", "NA", "n/a", "na", "--", "nan"], index_col=False)

    labelencoder = preprocessing.LabelEncoder()

    df['Gender'] = labelencoder.fit_transform(df['Gender'])

    oneHotCols = ["Card_Category", "Marital_Status"]

    df = pd.get_dummies(df, columns=oneHotCols)

    replaceStruct = {
        "Attrition_Flag": {"Existing Customer": 0, "Attrited Customer": 1},
        "Education_Level": {"Doctorate": 5, "Post-Graduate": 4, "Graduate": 3, "College": 2, "High School": 1,
                            "Unknown": 0, "Uneducated": -1},
        "Income_Category": {"$120K +": 4, "$80K - $120K": 3, "$60K - $80K": 2, "$40K - $60K": 1, "Unknown": 0,
                            "Less than $40K": -1}
    }

    df = df.replace(replaceStruct)

    return df


def feature_engineering_job():

    os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:8000"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Pablo"

    client = mlflow.tracking.MlflowClient()
    mlflow.set_experiment("bankchurners_feature_Engineering")

    with mlflow.start_run() as run:

        dataFrame = Feature_engineering(location="/Users/salv91/Desktop/mlflow_artifactRoot/mlruns/1/dd03e24808ee404296c929ee7c13a63b/artifacts/Raw_Data/Raw_Data_ek43an8g.csv")

        temp = tempfile.NamedTemporaryFile(prefix="Feature_Data_", suffix=".csv")
        temp_name = temp.name

        try:
            dataFrame.to_csv(temp_name, index=False)
            mlflow.log_artifact(temp_name, "Feature_Data")
        finally:
            temp.close()  # Delete the temp file

        # print info run
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        print(run.info)
        mlflow.end_run()