
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#test
def feature_engineering_job(location):

    """Feature Engineering for inference

    Returns:
        X_eval: Features
        Y_eval: Ground truth lables

    """
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

    # Train/test Split

    X = df.drop("Attrition_Flag", axis=1)
    y = df.pop("Attrition_Flag")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1, stratify=y)

    return X_test,y_test


