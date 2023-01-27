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
from sklearn.metrics import log_loss, accuracy_score

#load validation and train data
x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')

# Load the pre-trained model
with open("model1.sav", "rb") as f:
    model1 = pickle.load(f)
with open("model2.sav", "rb") as f:
    model2 = pickle.load(f)
with open("model3.sav", "rb") as f:
    model3 = pickle.load(f)
        
with open("model4.sav", "rb") as f:
    model4 = pickle.load(f)
with open("model5.sav", "rb") as f:
    model5 = pickle.load(f)    
    
    
models = [model1,model2,model3,model4,model5]

preds = [model.predict_proba(x_vali) for model in models]
preds = np.array(preds)
summed = np.sum(preds,axis = 0)


#argmax across classses
ensemble_prediction = np.argmax(summed,axis = 1) 

prediction1 = model1.predict(x_vali)
prediction2 = model2.predict(x_vali)
prediction3 = model3.predict(x_vali)
prediction4 = model4.predict(x_vali)
prediction5 = model5.predict(x_vali)

accuracy1 = accuracy_score(y_vali, prediction1)
accuracy2 = accuracy_score(y_vali, prediction2)
accuracy3 = accuracy_score(y_vali, prediction3)
accuracy4 = accuracy_score(y_vali, prediction4)
accuracy5 = accuracy_score(y_vali, prediction5)

ensemble_accuracy = accuracy_score(y_vali, ensemble_prediction)

print('Accuracy Score for model1 = ', accuracy1)
print('Accuracy Score for model2 = ', accuracy2)
print('Accuracy Score for model3 = ', accuracy3)
print('Accuracy Score for model4 = ', accuracy4)
print('Accuracy Score for model5 = ', accuracy5)
print('Accuracy Score for average ensemble = ', ensemble_accuracy)


#####################################################################
#Weighted average ensemble
models = [model1, model2, model3, model4,model5]
preds = [model.predict_proba(x_vali) for model in models]
preds=np.array(preds)
weights = [0.2, 0.2, 0.2,0.2,0.2]

#Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=1)

weighted_accuracy = accuracy_score(y_vali, weighted_ensemble_prediction)

print('Accuracy Score for model1 validation= ', accuracy1)
print('Accuracy Score for model2 validation= ', accuracy2)
print('Accuracy Score for model3 validation= ', accuracy3)
print('Accuracy Score for model4 validation= ', accuracy4)
print('Accuracy Score for model5 validation= ', accuracy5)
print('Accuracy Score for average ensemble = ', ensemble_accuracy)
print('Accuracy Score for weighted average ensemble = ', weighted_accuracy)

print()



#####################################################################


# Compute the predictions for each model
predictions_1 = model1.predict_proba(x_vali)
predictions_2 = model2.predict_proba(x_vali)
predictions_3 = model3.predict_proba(x_vali)
predictions_4 = model4.predict_proba(x_vali)
predictions_5 = model5.predict_proba(x_vali)

#find logloss for models
logloss1 = log_loss(y_vali,predictions_1)
logloss2 = log_loss(y_vali,predictions_2)
logloss3 = log_loss(y_vali,predictions_3)
logloss4 = log_loss(y_vali,predictions_4)
logloss5 = log_loss(y_vali,predictions_5)

ensemble_logloss_final = 1
ensemble_weights = []


#run the following part to find the optimal weights for models

for i in range (0,10):
    for j in range (0,10):
        for k in range (0,10):
            
            for l in range (0,10):
                for m in range (0,10):
                    sum_ijk = i + j + k + l + m + 1e-15
                    
                    weights[0] = i/sum_ijk
                    weights[1] = j/sum_ijk
                    weights[2] = k/sum_ijk
                    weights[3] = l/sum_ijk
                    weights[4] = m/sum_ijk
        
                    # Combine the predictions using the specified weights
                    ensemble_predictions = weights[0] * predictions_1 + weights[1] * predictions_2 + weights[2] * predictions_3 + weights[3] * predictions_4 + weights[4] * predictions_5
                    
                    # Compute the logloss of the ensemble predictions
                    ensemble_logloss = log_loss(y_vali, ensemble_predictions)
             
                    
                    #asssign new best score and weights
                    if ensemble_logloss < ensemble_logloss_final:
                        ensemble_logloss_final = ensemble_logloss
                        ensemble_weights = weights.copy()
                        print(f'Change in weights, new weights: \n {ensemble_weights} ')
        print(f'number of iterations = {i*10000+ j*1000}')

print()
print("model 1 logloss:", logloss1)
print("model 2 logloss:", logloss2)
print("model 3 logloss:", logloss3)
print("model 4 logloss:", logloss4)
print("model 5 logloss:", logloss5)
print("Ensemble logloss:", ensemble_logloss_final)







