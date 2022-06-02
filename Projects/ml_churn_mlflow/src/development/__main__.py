import etl
import feature_engineering
import promotion
import registering
import training

def main():

    etl.etl_job()
    feature_engineering.feature_engineering_job()
    training.training_job()
    registering.registering_job()
    promotion.promotion_job()

if __name__ == '__main__':
    main()