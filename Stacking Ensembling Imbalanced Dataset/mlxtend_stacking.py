from mlxtend.plotting import plot_confusion_matrix
from mlxtend.classifier import StackingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import log_loss,accuracy_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import time


x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')



model1 = XGBClassifier(objective='multi:softprob',
                      learning_rate = 0.1,
                      max_depth = 6,
                      n_estimators = 670,
                      subsample = 0.8,
                      colsample_bytree = 0.3)



model2 = XGBClassifier(objective='multi:softprob',
                      learning_rate = 0.1,
                      max_depth = 6,
                      n_estimators = 1000,
                      subsample = 0.2,
                      colsample_bytree = 0.3)


model3 = XGBClassifier(objective='multi:softprob',
                      learning_rate = 0.1,
                      max_depth = 5,
                      n_estimators = 1000,
                      subsample = 0.8,
                      colsample_bytree = 0.8,
                      reg_lambda = 1)

model4 = MLPClassifier(activation='relu',
                      random_state=50,
                      max_iter = 20,
                      hidden_layer_sizes=(128,),
                      solver='adam',
                      batch_size=128, 
                      alpha=0.001)
model5 = 



models = [model1, model2, model3, model4]    
logloss_models = dict()
#accuracies of the models
'''
i = 1
for model in models:
    time1 = time.time()
    model.fit(x_train,y_train)
    y_pred1 = model.predict_proba(x_vali)
    logloss1 = log_loss(y_vali,y_pred1)
    logloss_models[model] = logloss1
    time2 = time.time()
    print(f'For model{i}\n logloss = {logloss1} \n time = {time2-time1}')
'''    

time1 = time.time()
LogReg = LogisticRegression(max_iter=1000)
clf_stacking = StackingClassifier(models, 
                                  meta_classifier = LogReg,
                                  use_probas=True,
                                  use_features_in_secondary=True)
clf_stacking.fit(x_train,y_train)
time2 = time.time()

pred_meta = clf_stacking.predict_meta_features(x_vali)
y_pred = clf_stacking.predict_proba(x_vali)



Logloss1 = log_loss(y_vali,y_pred)
print(f'Logloss = {Logloss1} in {time2-time1}')














