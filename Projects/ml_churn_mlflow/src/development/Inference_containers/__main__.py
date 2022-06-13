import inference
import feature_engineering


def main():

    location_raw_data = "./data/Raw/BankChurners.csv"

    X_test,y_test = feature_engineering.feature_engineering_job(location_raw_data)
    print(f"Features is done")

    #training

    model_location = "./data/models/model.sav"
    inference.Inference_job(model_location,X_test,y_test)  
    print(f"Predictions are done")

if __name__ == '__main__':
    main()