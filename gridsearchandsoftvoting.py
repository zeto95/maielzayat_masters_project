import pandas as pd 
import warnings
import os
import Json
import tensorflow as tf
import numpy as np
import copy 
import time 
import pickle 
from sklearn.metrics import accuracy_score
from feature_extraction import Feature_Extractor 
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.models import Sequential 
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")
start_time = time.time()
print("############### Gradient search and voting script starting time ###############",start_time)


#load data 
dataset = pd.read_csv('Training_features.txt',sep= '\t', encoding="ISO-8859-1")
# dataset= dataset.dropna()
dict = {'Coorleation': 0, 'Descriptive_stats': 1, 'Model_results': 2, 'Variable_def':3, 'others':4}
X = dataset.iloc[:, 1:10]
Y = dataset['Label'].apply(lambda label: dict[label])

"""split data into training and testing data with test size of 20%"""
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)


 
"""Create keras neural network model"""
# opt = keras.optimizers.Adam(learning_rate=0.01) 
input_dim = len(X.columns)
def create_model(dropout_rate = 0.2, optimizer = 'Adam'):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

""" Define a pipeline for Gridsearch"""
def algorithm_pipeline(train_x, test_x, train_y, test_y, model, param_grid ,cv=10 , do_probabilities = False):
    
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=['accuracy'],
        verbose=0, 
        refit = 'accuracy'
    )
    fitted_model = gs.fit(train_x, train_y)
    
    if do_probabilities:
      pred = fitted_model.predict_proba(test_x)
    else:
      pred = fitted_model.predict(test_x)
    
    return fitted_model, pred

""" Gridsearch for Keras deep learning model"""
keras_param_grid = {
              'batch_size':[128,256],
              'epochs' :              [100,150,200,300,400,500],
            #  'batch_size' :          [32, 128],
              'optimizer' :           ['Adam', 'Nadam','SGD'],
              'dropout_rate' :        [0.1,0.2, 0.3]
            #  'activation' :          ['relu', 'elu']
             }
Keras_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model,  verbose=False)
Keras_model._estimator_type = 'classifier'
Keras_model, pred = algorithm_pipeline(train_x, test_x, train_y, test_y, Keras_model, keras_param_grid,cv=5)

"""Saving Keras best parameters as pickle file"""
keras_best_pars = Keras_model.best_params_
pickle.dump(Keras_model.best_params_, open("keras_log_reg.pickle", "wb"))

print("Keras model parameters saved...")

""" Gridsearch for XGBoost model"""
xgb_model = XGBClassifier()
xgb_param_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9]
}
xgb_model, pred = algorithm_pipeline(train_x, test_x, train_y, test_y, xgb_model, xgb_param_grid,cv=5)

"""Saving XGBoost best parameters as pickle file"""
xgb_best_pars = xgb_model.best_params_
pickle.dump(xgb_model.best_params_, open("xgb_log_reg.pickle", "wb"))

print("XGB model parameters saved...")


""" Grid search for RFC model"""
rfc_model = RandomForestClassifier()
rfc_param_grid = {
'n_estimators': [400, 700, 1000],
    'max_depth': [15,20,25],
    'max_leaf_nodes': [50, 100, 200]
}
rfc_model, pred = algorithm_pipeline(train_x, test_x, train_y, test_y, rfc_model, 
                                 rfc_param_grid,cv=5)
"""Saving RFC best parameters as pickle file"""
rfc_best_pars = rfc_model.best_params_
pickle.dump(rfc_model.best_params_, open("RFC_log_reg.pickle", "wb"))

print("RFC model parameters saved...")
print("############# GridSearch finished, Voting Started #############")

""" Ensemble voting classifier"""
eclf = VotingClassifier(
    estimators=[('Keras', Keras_model), 
                ('XGB', xgb_model), 
                ('RFC', rfc_model)],   
                voting= 'soft',
                flatten_transform=False)

eclf.fit(train_x, train_y)


print("Keras Best score", Keras_model.best_score_)
print("Keras Best Params" , Keras_model.best_params_)
print("XGboost Best score" , np.sqrt(xgb_model.best_score_))
print("Xgboost Best params" , xgb_model.best_params_)
print("RFC Best score" , rfc_model.best_score_)
print("RFC Best params" , rfc_model.best_params_)
print("Best Estimator",)


print('5-fold cross validation:\n')

for clf in (Keras_model, xgb_model, rfc_model, eclf):
    y_pred = clf.predict(test_x)
    print("Model after voting ",clf.__class__.__name__, accuracy_score(test_y, y_pred))


end_time = time.time()
print("############### Gradient search and voting script end time ###############",end_time)


delta = end_time-start_time
print("############### Time taken for gridsearch and voting ###############",delta)