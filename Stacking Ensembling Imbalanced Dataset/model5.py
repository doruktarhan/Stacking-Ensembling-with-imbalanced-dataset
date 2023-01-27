import numpy as np
import pandas as pd
import time
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import SCORERS
import pickle

x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')






val_set =(x_vali,y_vali)

model = RandomForestClassifier(n_estimators=5000,
                               max_features= 'auto',
                               min_samples_split = 2,
                               min_samples_leaf = 1,
                               bootstrap=False,
                               random_state = 42,
                               verbose = True,
                               n_jobs=-1)

params = model.get_params()
model.fit(x_train,y_train)

y_pred_train = model.predict_proba(x_train)
y_pred = model.predict_proba(x_vali)
train_loss = log_loss(y_train,y_pred_train)
logloss = log_loss(y_vali,y_pred)


filename = 'model5.sav'
pickle.dump(model, open(filename, 'wb'))


