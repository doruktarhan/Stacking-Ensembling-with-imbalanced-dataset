import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

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

sfm_selector = SelectFromModel(trained_model,prefit = True )
selected_features_bool = sfm_selector.get_support()


#count = np.count_nonzero(selected_features_bool)
#features = sfm_selector.transform(features)

print(f'New shape of the features = {features.shape}')




treshold1 = 1
#find the treshold 
for i in range(len(feature_imp)):
    if selected_features_bool[i] and treshold1 > feature_imp[i]:
        treshold1 = feature_imp[i]
        
x_train,x_vali,y_train,y_vali = train_test_split(features,lc_y,random_state=50)


#PARAMETERS FOR START
#n_estimators = number of trees / optimal found 500 trees
#max_depth = depth of trees / optimal 3-4 
#eta (learning_rate for xgbclassifier) = learning rate of the trees optimal 0.1
#subsample = fractipon of the dataset to be used / 0.8 optimal 
#colsample_bytree = portiopn of columns by a tree / optimal 0.6 and above


model = XGBClassifier(objective='multi:softprob',
                      learning_rate = 0.06,
                      max_depth = 6,
                      n_estimators = 2000,
                      subsample = 0.8,
                      colsample_bytree = 0.8
                      )


eval_set1 = [(x_train, y_train), (x_vali, y_vali)]
time1 = time.time()
model.fit(x_train,
          y_train,
          eval_set = eval_set1,
          eval_metric = 'mlogloss',
          early_stopping_rounds = 100)
time2 = time.time()
print(f'time for fitting the model {time2-time1}')



results = model.evals_result()

#plot the results on the grid 
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.xlabel('\nEpochs',fontsize=14,fontweight='semibold',color='white')
plt.ylabel('Error\n',fontsize=14,fontweight='semibold',color='white')
plt.title('XGBoost learning curve\n',fontsize=20,fontweight='semibold',color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()




