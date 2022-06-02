######## Libraries & Modules ##########
import os
import Projects.ml_churn_mlflow.src.utils.helpers.postgressql as p_sql
import mlflow
import tempfile
############################################


def etl_job():

    conn_params_dic = {
        "host"      : "localhost",
        "database"  : "bank_raw_data",
        "user"      : "mlflow_user"
    }

    os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:8000"
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Pablo"

    client = mlflow.tracking.MlflowClient()

    mlflow.set_experiment("bankchurners_etl")

    with mlflow.start_run() as run:

        connection = p_sql.connect(conn_params_dic)

        column_names = ['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
                        'Dependent_count', 'Education_Level', 'Marital_Status',
                        'Income_Category', 'Card_Category', 'Months_on_book',
                        'Total_Relationship_Count', 'Months_Inactive_12_mon',
                        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

        table_name = "users"

        df = p_sql.query_read(connection, column_names=column_names, table_name=table_name)

        temp = tempfile.NamedTemporaryFile(prefix="Raw_Data_", suffix=".csv")
        temp_name = temp.name

        try:
            df.to_csv(temp_name, index=False)
            mlflow.log_artifact(temp_name, "Raw_Data")
        finally:
            temp.close()  # Delete the temp file

        # print info run
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        print(run.info)
        mlflow.end_run()