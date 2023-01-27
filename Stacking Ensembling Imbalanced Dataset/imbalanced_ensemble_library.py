from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier,RUSBoostClassifier,EasyEnsembleClassifier
import numpy as np
from sklearn.metrics import log_loss,accuracy_score



x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')

classifier = EasyEnsembleClassifier(random_state= 0)
classifier.fit(x_train,y_train,)


y_pred_acc =classifier.predict(x_vali)
acc = accuracy_score(y_vali, y_pred_acc)
y_pred = classifier.predict_proba(x_vali)
logloss= log_loss(y_vali,y_pred)
print(acc,logloss)