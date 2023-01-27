import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv('train_data_swc.csv')

features = np.array(df.iloc[:,:-1])
labels = np.array(df[['y']]).ravel()

#label encoder for xgboost
lc = LabelEncoder() 
lc = lc.fit(labels) 
lc_y = lc.transform(labels)


filename = 'finalized_model.sav'
trained_model = pickle.load(open(filename, 'rb'))
feature_imp = trained_model.feature_importances_



sfm_selector = SelectFromModel(XGBClassifier(objective='multi:softprob',
                      learning_rate = 0.1,
                      max_depth = 5,
                      n_estimators = 1000,
                      subsample = 0.8,
                      colsample_bytree = 0.8)
)
sfm_selector.fit(features,lc_y)
selected_features_bool = sfm_selector.get_support()


count = np.count_nonzero(selected_features_bool)
features = sfm_selector.transform(features)


print(f'New shape of the features = {features.shape}')

x_train,x_vali,y_train,y_vali = train_test_split(features,lc_y,random_state=50)


model = XGBClassifier(objective='multi:softprob',
                      learning_rate = 0.1,
                      max_depth = 5,
                      n_estimators = 1000,
                      subsample = 0.8,
                      colsample_bytree = 0.8)

#define parameters for randomized searh cv
params = {
    'n_estimators' : [1000],
    'max_depth' : [4,5,6],
    'learning_rate' : [0.05,0.1,0.2],
    'subsample' : [0.5,0.8],
    'colsample_bytree' : [0.5,0.8,1]
    }


eval_set1 = [(x_train, y_train), (x_vali, y_vali)]



rand_search = RandomizedSearchCV(model ,params,scoring='neg_log_loss',n_iter = 20,verbose =3,cv = 4)
rand_search.fit(x_train,y_train)


results = rand_search.cv_results_
result_params = results['params']
result_scores = results['mean_test_score']


for i in range (20):
    print(f'for the parameters {result_params[i]} the score of cross validation is {result_scores[i]}')


#modeli en son tekrar trainle yeni parametrelerle olusturup sonra da x_test,y_test yap

sort_index_features = np.argsort(feature_imp)
sorted_features = feature_imp[sort_index_features]
