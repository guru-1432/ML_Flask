import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

#Loading the Dataset
def load_data_set():
    iris = load_iris()

# Extraction column names.
    
    targe_names = [i for i in iris.target_names]
    feature_names = [i for i in iris.feature_names]

# Creating dataf frame with data and the target 
    dataframe = pd.DataFrame(iris.data,columns=feature_names)
    target_df = pd.DataFrame(iris.target,columns=['Target']) 

# split the dataset for training and testing
    return (train_test_split(dataframe,np.ravel(target_df),test_size = 0.2))

def Train_Model():
    x_train,x_test,y_train,y_test =  load_data_set() 
    
    #SVR Support vector regression
    model_svr = SVR()
    model_svr.fit(x_train,y_train)
    svr_score = (model_svr.score(x_test,y_test))
    
    #SVC - support vector classification
    model_svc = SVC(kernel = 'poly',C=10,gamma= 1)
    model_svc.fit(x_train,y_train)
    svc_score = (model_svc.score(x_test,y_test))
    #Logistice Regression/SVC
    model_LR = LogisticRegression()
    model_LR.fit(x_train,y_train)
    model_LR.score(x_test,y_test)
    LR_score = (model_LR.score(x_test,y_test))

    with open ('Train_Model','wb') as f:
        pickle.dump([model_svc,model_svc,model_LR],f)
        
#Logistice Regression/SVC
def create_mdoel():
    if not os.path.isfile('Train_Model'):
        Train_Model()
        return 'Creating the Model'    
    else:
        return 'Model already exist'


def create_mdoel():
    if not os.path.isfile('Train_Model'):
        Train_Model()
        return 'Creating the Model'    
    else:
        return 'Model already exist'

def create_mdoel():
    if not os.path.isfile('Train_Model'):
        Train_Model()
        return 'Creating the Model'    
    else:
        return 'Model already exist'






