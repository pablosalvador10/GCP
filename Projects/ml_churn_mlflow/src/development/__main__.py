import etl
import feature_engineering
import promotion
import registering
import training

def main():

    ## Set up connection params

    conn_params_dic = {
        "host": "localhost",
        "database": "bank_raw_data",
        "user": "mlflow_user"
    }

    location_raw_data = etl.etl_job(conn_params_dic)
    print(f"LOCATION RAW DATA {location_raw_data}")

    location_features = feature_engineering.feature_engineering_job(location_raw_data)
    print(f"LOCATION FEATURES DATA {location_features}")

    #training
    training.training_job(location_features)

    #Registering Model
    registering.registering_job()

    #Promotion Model to Prod
    promotion.promotion_job()

if __name__ == '__main__':
    main()