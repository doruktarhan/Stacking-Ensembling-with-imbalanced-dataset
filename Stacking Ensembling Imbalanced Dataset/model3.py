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
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score




#GET DATA / FEATURE ENGINEERING

x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')


############################################################################3

val_set =(x_vali,y_vali)

# Create an instance of the MLPClassifier with the desired hyperparameters
model = MLPClassifier(activation='relu',
                      random_state=50,
                      verbose=True,
                      max_iter = 20,
                      hidden_layer_sizes=(128,),
                      solver='adam',
                      batch_size=128, 
                      alpha=0.001)

model.fit(x_train,
          y_train
          )




y_pred =model.predict_proba(x_vali)
logloss = log_loss(y_vali,y_pred)


filename = 'model3.sav'
pickle.dump(model, open(filename, 'wb'))
