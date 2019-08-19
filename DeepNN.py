#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:58:11 2019

@author: sachin
"""

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from keras.models import load_model
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

class DeepNN():
    def __init__(self, ki, opt, loss, batch_size, epochs, X, y, i_dims, X_test):
        self.kernel_initializer = ki
        self.optimizer = opt
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.X = X
        self.y = y
        self.i_dim = i_dims
        self.X_test = X_test
        
    def initialize(self):
        self.classifier = Sequential()
    
    def input_layer(self, units):
        self.classifier.add(Dense(units = units,
                                  kernel_initializer = self.kernel_initializer,
                                  use_bias = False,
                                  input_dim = self.i_dim))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Activation('relu'))
        
        
    def hidden_layer(self, units):
        self.classifier.add(Dense(units = units,
                                  kernel_initializer = self.kernel_initializer,
                                  use_bias = False))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Activation('relu'))
        
    def output_layer(self):
        self.classifier.add(Dense(units = 1,
                                  kernel_initializer = self.kernel_initializer,
                                  use_bias = False))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Activation('sigmoid'))
    
    def optimize(self):
        self.classifier.compile(optimizer = self.optimizer,
                                loss = self.loss,
                                metrics = ['accuracy'])
        
    def train(self):
        self.classifier.fit(self.X, self.y,
                            batch_size = self.batch_size,
                            epochs = self.epochs)
    
    def predict(self):
        y_pred = self.classifier.predict(self.X_test)
        return y_pred
    
    def model(self):
        self.initialize()
        self.input_layer(200)
        self.hidden_layer(100)
        self.hidden_layer(50)
        self.hidden_layer(25)        
        self.output_layer()
        self.optimize()
        self.train()
        y_pred = self.predict()
        return y_pred, self.classifier
    
    def model_accuracy(self):      # for finding the Mean accuracy.
        self.initialize()
        self.input_layer(200)
        self.hidden_layer(100)
        self.hidden_layer(50)
        self.hidden_layer(25)
        self.output_layer()
        self.optimize()
        return self.classifier
    
    def model_grid_search(self):   # for grid search.
        self.initialize()
        self.input_layer(200)
        self.hidden_layer(100)
        self.hidden_layer(50)
        self.hidden_layer(25)
        self.output_layer()
        self.optimize()        
        return self.classifier

############################ Pre-Processing Step ###############################
 
dataset = pd.read_csv('bank-full.csv',
                      sep = ";",
                      encoding = "latin-1",
                      engine = 'python')

X = dataset.iloc[:,[0,2,3,4,5,6,7,11,12,14]].values
#X = np.array(X, dtype = np.float64)
y = dataset.iloc[:,-1].values

label_x = LabelEncoder()
X[:, 3] = label_x.fit_transform(X[:, 3])

label_x1 = LabelEncoder()
X[:, 5] = label_x1.fit_transform(X[:, 5])

label_x2 = LabelEncoder()
X[:, 6] = label_x2.fit_transform(X[:, 6])

label_x3 = LabelEncoder()
X[:, 1] = label_x3.fit_transform(X[:, 1])

label_x4 = LabelEncoder()
X[:, 2] = label_x4.fit_transform(X[:, 2])

one_hot_x3 = OneHotEncoder(categorical_features = [1])
X = one_hot_x3.fit_transform(X).toarray()
X = X[:, 1:]

im_x = Imputer(missing_values = 3, strategy = "most_frequent", axis = 0)
X[:, [3]] = im_x.fit_transform(X[:, [3]])
 
one_hot_x4 = OneHotEncoder(categorical_features = [3])
X = one_hot_x4.fit_transform(X).toarray()
X = X[:, 1:]

y = np.reshape(y, (-1, 1))
label_y = LabelEncoder()
y[:, 0] = label_y.fit_transform(y[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.05)

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

input_dims = X_train.shape[1]

########################### Initializing the Model #############################

result = DeepNN(ki = 'glorot_normal',
                opt = 'adam',
                loss = 'binary_crossentropy',
                batch_size = 512,
                epochs = 500,
                X = X_train,
                y = y_train,
                i_dims = input_dims,
                X_test = X_test)

############################ predicting the Model ##############################

predictions, classifier = result.model()        
predictions = (predictions > 0.5)   

y_test = np.array(y_test, dtype = np.bool)  
cm = confusion_matrix(y_test, predictions)    

from sklearn.metrics import accuracy_score
accuracy_test_set = accuracy_score(y_test, predictions)
     
########################### K-Fold Cross Validation ############################

classifier1 = KerasClassifier(result.model_accuracy,
                              batch_size = 512,
                              epochs = 500)

accuracies = cross_val_score(estimator = classifier1,
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             n_jobs = -1)   

accuracy_mean = accuracies.mean()   
accuracy_variance = accuracies.std()    
    
################# Grid-Search for finding the best parameters ##################

classifier2 = KerasClassifier(result.model_grid_search,
                              batch_size = 512)

parameters = {'epochs' : [100, 300, 500],
              'batch_size' : [128, 256, 512]}

grid_search = GridSearchCV(estimator = classifier2,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)   

y_train = np.array(y_train, dtype = np.int)    
grid_search = grid_search.fit(X_train, y_train)    
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_   
    
     
############################### Saving the model ################################    
    
classifier.save('model.h5')
del classifier

##################### Loading and visualising the model #########################

classifier = load_model('model.h5')   
plot_model(classifier, to_file= 'model.png')    
SVG(model_to_dot(classifier).create(prog='dot', format='svg'))    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 