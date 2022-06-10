import os
import pandas as pd
import mlflow
import tempfile
from sklearn import preprocessing

#test
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


def feature_engineering_job(location):

    os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:8000"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Pablo"

    client = mlflow.tracking.MlflowClient()
    mlflow.set_experiment("bankchurners_feature_Engineering")

    with mlflow.start_run() as run:

        df = Feature_engineering(location=location)

        temp_dir = tempfile.gettempdir()
        new_data = f"{temp_dir}" + "/features_data.csv"
        print(temp_dir)
        try:
            df.to_csv(new_data, index=False)
            mlflow.log_artifact(new_data, "Features_Data")
        finally:
            pass
            # tempfile.close()  # Delete the temp file

        # Print Info Run

        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("raw_data", location)
        mlflow.set_tag("artifact_uri", artifact_uri)

        s = f"{artifact_uri}/" + f"Features_Data/" + "features_data.csv"

        #print(run.info)
        #print(s)
        mlflow.end_run()

        return s