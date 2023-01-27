import lightgbm as lgb
import numpy as np
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.multiclass import _ConstantPredictor
from sklearn.preprocessing import LabelBinarizer
from scipy import special
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss




x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')



'''
fit = lgb.Dataset(x_train, y_train)
val = lgb.Dataset(x_vali, y_vali, reference=fit)

model = lgb.train(
    params={
        'learning_rate': 0.01,
        'objective': 'multiclass',
        'metric':'multi_logloss',
        'num_class' : 9
    },
    train_set=fit,
    num_boost_round=10000,
    valid_sets=(fit, val),
    valid_names=('train', 'val'),
    early_stopping_rounds=20,
    verbose_eval=100,
)

params={
    'learning_rate': 0.01,
    'objective': 'multiclass',
    'metric':'multi_logloss',
    'num_class' : 9,

}
'''
'''
Parameter Guide https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5
USE MANY DECISION TREES AND DECREASE THE LEARNING RATE
FIND BEST COMBINATION OF n_estimators and learning_rate

num_leaves: (default = 31)
    Controls number of leaves in a single tree 
    num_leaves should be 2^max_depth
    more effective in learning than max depth 
    should search(20-3000)

max_depth: (default = -1)
    Max tree limit for base learners 
    Should searxh (3-12)
    
learning_rate: (deafult = 0.1)
    
lambda_l1 and lambda_l2:
    good search (0-100)
    
min_gain_to_split: 
    similar to xgboost gamma
    conservative search in range (0,15)
    
bagging_fraction feature_fraction falan da var


 # "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
 # "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
 # "max_depth": trial.suggest_int("max_depth", 3, 12),
 # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
 # "max_bin": trial.suggest_int("max_bin", 200, 300),
 # "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
 # "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
 # "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
 # "bagging_fraction": trial.suggest_float(
 #     "bagging_fraction", 0.2, 0.95, step=0.1
 # ),
 # "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
 # "feature_fraction": trial.suggest_float(
 #     "feature_fraction", 0.2, 0.95, step=0.1

'''

model2 = LGBMClassifier(objective = 'multiclass',
                        n_estimators=10000,
                        num_leaves=100,
                        learning_rate=0.01,
                        max_depth = 6
                        )
model2.fit(
    x_train,
    y_train,
    eval_set=[(x_vali, y_vali)],
    eval_metric="multi_logloss",
    early_stopping_rounds=100
    )
    
y_pred = model2.predict_proba(x_vali)
logloss = log_loss(y_vali,y_pred)
