# Stacking-Ensembling-with-imbalanced-dataset


In this project I have used lots of classification methods such as XGboost, lightgbm, ANN, SVM and then calculated the perfomances of them. 

The main goal is to decreases the logloss error as much as possible 

Data consists of 9 class and 60k samples with more than 100 faeatures so preproceessing must be done before training also. 

After the preproccesing is done, I have splitted the data as train and validation sets by arranging the class numbers since the data has imbalanced dataset. The preprocessing is odne in feature_eng.py

Hyperparameter tuning is done for all of the models. Hyperparameter tuninig takes too much time especially if you use GridSearchCV so that alternative methods can be used such as optuna. 

Customized loss function for imbalanced dataset was made for better results in ONEvsRestLightGBMCustomizedLoss.py

Also Max_Halford Focal loss was tried but it did not work well. 


Some models form imbalanced_ensemble library also tested but also did not give good results with the dataset. 

