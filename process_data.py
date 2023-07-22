from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Probability Ratio encoding for categorical feature EJ
def PRE(val, data):
    
    t = len(data[(data['EJ'] == val) & (data['Class'] == 1)])
    f = len(data[(data['EJ'] == val) & (data['Class'] == 0)])
    if f:
        return t/f
    return 1

# Count/Frequency encoding for categorical features in multi-class classification
def CFE(val, feature, data):
    count = len(data[data[feature] == val])
    print(f"{feature}: {val} -> {count}")
    return count
    
# Normalizing data (Note that we scale the training and test set separately)
def Normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def SplitShapeData(X, y):
    # Obtaining test and training sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,shuffle=False)

    # Replace missing values with 0 and convert dataframe to numpy array
    X_train = np.nan_to_num(np.array(Normalize(X_train)))
    X_test = np.nan_to_num(np.array(Normalize(X_test)))
    y_train = np.array(y_train).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)

    return X_train,X_test,y_train,y_test
    
def ProcessData():
    data = pd.read_csv("train.csv")

    # Obtain X and y
    X = data.iloc[:,1:57]
    y = data.iloc[:,57]

    print(X.isnull().sum())
    
    X = X.replace('A', PRE('A', data))
    X = X.replace('B', PRE('B', data))
    
    return SplitShapeData(X, y)

def getMultiClassData():
    data = pd.read_csv("train.csv")
    greeks = pd.read_csv("greeks.csv")
    
    classes = {'A': 0, 'B': 1, 'D': 2, 'G': 3}
    # Obtain X and y
    X = data.iloc[:,1:57]
    y = greeks.iloc[:,1]
    
    for c in classes:
        y = y.replace(c, classes[c])
        
        
    # Using One Hot encoding since there are only two nomial categories for EJ
    # i.e. will not increase dimensionality    
    EJ_indicator = pd.get_dummies(data['EJ'], prefix="EJ: ", drop_first=True)
    
    X = pd.concat([X, EJ_indicator], axis=1)
    X.drop('EJ', axis=1, inplace=True)
    
    return SplitShapeData(X, y)

def getExperimentalData():
    data = pd.read_csv("train.csv")
    greeks = pd.read_csv("greeks.csv")
    
    # Obtain X and y
    X = data.iloc[:,1:57]
    y = greeks.iloc[:,1]
    experimental = greeks.iloc[:,2:5]

    X = pd.concat([X, experimental], axis=1)

    y = y.replace({'A': 0, 'B': 1, 'D': 2, 'G': 3})
    data = pd.concat([data, greeks], axis=1)
    
    beta_dict = {x: CFE(x, 'Beta', data) for x in data['Beta'].unique()}
    delta_dict = {x: CFE(x, 'Delta', data) for x in data['Delta'].unique()}
    gamma_indicators = pd.get_dummies(data['Gamma'])
    EJ_indicator = pd.get_dummies(data['EJ'], prefix="EJ: ", drop_first=True)
    
    print(gamma_indicators)
    print(EJ_indicator)
    
    X = X.replace({'Beta': beta_dict, 'Delta': delta_dict})
    X = pd.concat([X, gamma_indicators], axis=1)
    X = pd.concat([X, EJ_indicator], axis=1)
    X.drop('Gamma', axis=1, inplace=True)
    X.drop('EJ', axis=1, inplace=True)
    
    return SplitShapeData(X, y)

X_train,_,_,_ = getMultiClassData()
print(X_train.shape)