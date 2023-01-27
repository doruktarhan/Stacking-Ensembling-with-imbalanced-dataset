import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import SCORERS
import pickle
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import seaborn as sns





##################################################################
#GET DATA / FEATURE ENGINEERING

x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')


#oversampling with smote or randomsampler uncomment when needed
#smote = SMOTE(random_state = 0)    
#ros = RandomUnderSampler(random_state=0) 
#x_train,y_train = ros.fit_resample(x_train,y_train) 
#x_train,y_train = smote.fit_resample(x_train,y_train)

#weights for the classes in order to deal with imbalanced dataset 
#weight = CreateBalancedSampleWeights(y_train, largest_class_weight_coef)


##################################################################
#DEFINE MODEL AND TRAIN IT


#Xgboost with tuned hyperparameters
model = XGBClassifier(objective='multi:softprob',
                      learning_rate = 0.1,
                      max_depth = 5,
                      n_estimators = 1000,
                      subsample = 0.8,
                      colsample_bytree = 0.8,
                      early_stopping_rounds = 100,
                      reg_lambda = 1)




#define eval set for training
eval_set1 = [(x_train, y_train), (x_vali, y_vali)]

#train model
time1 = time.time()
model.fit(x_train,
          y_train,
          eval_set = eval_set1,
          eval_metric = 'mlogloss')
time2 = time.time()
print(f'time for fitting the model {time2-time1}')


#results for the model
results = model.evals_result()

#predict the model with validation set and define confusion matrix
y_pred = model.predict_proba(x_vali)

filename = 'model4.sav'
pickle.dump(model, open(filename, 'wb'))