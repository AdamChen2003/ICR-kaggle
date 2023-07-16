from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def ProcessData():
    data = pd.read_csv("train.csv")

    # Obtain X and y
    X = data.iloc[:,1:57]
    y = data.iloc[:,57]

    # Probability Ratio encoding for categorical feature EJ
    def PRE(val):
        t = len(data[(data['EJ'] == val) & (data['Class'] == 1)])
        f = len(data[(data['EJ'] == val) & (data['Class'] == 0)])
        if f:
            return t/f
        return 1

    X = X.replace('A', PRE('A'))
    X = X.replace('B', PRE('B'))

    print(X.isnull().sum())

    # Obtaining test and training sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,shuffle=False)

    # Normalizing data (Note that we scale the training and test set separately)
    def Normalize(X):
        scaler = StandardScaler().fit(X)
        return scaler.transform(X)

    # Replace missing values with 0 and convert dataframe to numpy array
    X_train = np.nan_to_num(np.array(Normalize(X_train)))
    X_test = np.nan_to_num(np.array(Normalize(X_test)))
    y_train = np.array(y_train).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)

    return X_train,X_test,y_train,y_test