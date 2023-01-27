import numpy as np
import pandas as pd
import time
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import SCORERS


x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')






val_set =(x_vali,y_vali)

# Number of trees in random forest
n_estimators = [400,600,800,1000,1300,1600,2400,3000,5000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [50,100,200,300,400,800]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [1,2,4,8,16,32,48]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model = RandomForestClassifier(random_state = 42)
model.get_params()

time1 = time.time()

rf_random = RandomizedSearchCV(estimator = model, 
                               param_distributions = random_grid,
                               scoring='neg_log_loss',
                               n_iter = 20, 
                               verbose=3, 
                               cv = 5,
                               random_state=42, 
                               n_jobs = -1,
                               return_train_score = True)

# Fit the random search model
rf_random.fit(x_train,
              y_train)

time2 = time.time()

'''
log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
time1 = time.time()

model = RandomForestClassifier(n_estimators= 2000,
                                max_depth=50,
                                max_features = 'auto',
                                min_samples_leaf=2,
                                min_samples_split=2)


a,train_loss,val_loss = learning_curve(model,
                        x_train,
                        y_train,
                        train_sizes=np.linspace(0.1, 1.0, 5),
                        scoring='neg_log_loss',
                        cv=5,
                        n_jobs = -1)


time2 = time.time()

y_pred = model.predict_proba(x_vali)
logloss=log_loss(y_vali, y_pred)
print(f'rime = {time2-time1} \n logloss = {logloss}')

'''



param_list = rf_random.cv_results_['params']
test_scores = rf_random.cv_results_['mean_test_score']
train_scores =rf_random.cv_results_['mean_train_score']



index_list = [2,10,14,17,0,1,4,5,7,8,9]

for i in index_list:
    print(f'{param_list[i]}\n train_score = {train_scores[i]} \n test_score = {test_scores[i]}')
    print()
















