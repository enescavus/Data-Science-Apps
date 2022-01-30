import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def data_load(print_df=False):
    '''
        Returns : X: training columns y: label columnd

        Creates : New dataset csv file into the Data folder
    '''
    X, y = load_iris().data, load_iris().target

    df = pd.concat([pd.DataFrame(data=X, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]), pd.DataFrame(data=y, columns=["label"]) ], axis=1)
    df = df.sample(frac=1)  # shuffle 
    df.to_csv("data/iris_data.csv")
    if print_df==True:
        print(df)

    return X,y

def data_split(X,y):
    '''
        Returns : Split data format for ML algorithms
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                shuffle=True,
                                                test_size=.2,
                                                stratify=y,
                                                random_state=0)
    return X_train, X_test, y_train, y_test
