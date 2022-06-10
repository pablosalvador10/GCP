######## Libraries & Modules ##########
import os
import Projects.ml_churn_mlflow.src.utils.helpers.postgressql as p_sql
import mlflow
import tempfile
############################################





def etl_job(params):


    os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:8000"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Pablo"

    client = mlflow.tracking.MlflowClient()

    mlflow.set_experiment("bankchurners_etl")

    with mlflow.start_run() as run:

        connection = p_sql.connect(params)

        column_names = ['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
                        'Dependent_count', 'Education_Level', 'Marital_Status',
                        'Income_Category', 'Card_Category', 'Months_on_book',
                        'Total_Relationship_Count', 'Months_Inactive_12_mon',
                        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

        table_name = "users"

        df = p_sql.query_read(connection, column_names=column_names, table_name=table_name)

        temp_dir = tempfile.gettempdir()
        new_data = f"{temp_dir}" + "/raw_data.csv"

        try:
            df.to_csv(new_data, index=False)
            mlflow.log_artifact(new_data, "Raw_Data")
        finally:
            pass
            #tempfile.close()  # Delete the temp file

        # Print Info Run

        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("artifact_uri", artifact_uri)

        s = f"{artifact_uri}/" + f"Raw_Data/" + "raw_data.csv"

        #print(run.info)
        #print(s)
        mlflow.end_run()


        return s