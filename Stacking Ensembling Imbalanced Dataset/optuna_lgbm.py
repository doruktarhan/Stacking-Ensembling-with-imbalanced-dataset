import lightgbm as lgb
import numpy as np
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.multiclass import _ConstantPredictor
from sklearn.preprocessing import LabelBinarizer
from scipy import special
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from optuna.integration import LightGBMPruningCallback

x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')


X = np.concatenate((x_train,x_vali),axis = 0)
y = np.concatenate((y_train,y_vali),axis = 0)

def objective(trial,X,y):
    param_grid = {
        #"device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [5000,10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
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
        #),
    }
    

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)
    cv_scores = np.empty(5)
    
    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LGBMClassifier(objective="multi", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="multi_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "multi_logloss")
            ],  # Add a pruning callback
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)

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


'''



study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
func = lambda trial: objective(trial,X,y)
study.optimize(func, n_trials=5)














