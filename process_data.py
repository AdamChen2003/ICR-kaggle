import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Probability Ratio encoding for categorical feature EJ
def PRE(val, data):
    
    t = len(data[(data['EJ'] == val) & (data['Class'] == 1)])
    f = len(data[(data['EJ'] == val) & (data['Class'] == 0)])
    if f:
        return t/f
    return 1

# Count/Frequency encoding for categorical features in multi-class classification
def CFE(val, feature, data):
    return len(data[data[feature] == val])
    
# Normalizing data
def Normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
    
def getBinaryClassData():
    data = pd.read_csv("train.csv")

    # Obtain X and y
    X = data.iloc[:,1:57]
    y = data.iloc[:,57]
    
    X = X.replace('A', PRE('A', data))
    X = X.replace('B', PRE('B', data))
    
    return np.array(X),np.array(y).reshape(-1,1)

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

    return np.array(X),np.array(y).reshape(-1,1)

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
    
    X = X.replace({'Beta': beta_dict, 'Delta': delta_dict})
    X = pd.concat([X, gamma_indicators], axis=1)
    X = pd.concat([X, EJ_indicator], axis=1)
    X.drop('Gamma', axis=1, inplace=True)
    X.drop('EJ', axis=1, inplace=True)
    
    return np.array(X),np.array(y).reshape(-1,1)

def getBinaryClassWeights():
    data = pd.read_csv("train.csv")
    neg, pos = np.bincount(data['Class'])
    total = neg + pos
    neg_weight = (1 / neg) * (total / 2.0)
    pos_weight = (1 / pos) * (total / 2.0)

    return {0:neg_weight, 1:pos_weight}

# def getMultiClassWeights():