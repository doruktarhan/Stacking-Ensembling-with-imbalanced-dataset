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

df = pd.read_csv('train_data_swc.csv')
df_test = pd.read_csv('test_data_swc.csv')

x_test = np.array(df_test)
features = np.array(df.iloc[:,:-1])
labels = np.array(df[['y']]).ravel()

#label encoder for xgboost
lc = LabelEncoder() 
lc = lc.fit(labels) 
lc_y = lc.transform(labels)


#scale the features
scaler = StandardScaler().fit(features)
features = scaler.transform(features)
x_test_transformed = scaler.transform(x_test)




#stratified class split for better learning
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(features, lc_y):
    x_train, x_vali = features[train_index], features[test_index]
    y_train, y_vali = lc_y[train_index], lc_y[test_index]


np.save('x_train.npy',x_train)
np.save('x_vali.npy',x_vali)
np.save('y_train.npy',y_train)
np.save('y_vali.npy',y_vali)
np.save('x_test.npy',x_test_transformed)