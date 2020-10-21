#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:26:41 2020

@author: jeffreysun
"""

# Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Reading in the dataset with pandas
dataset = pd.read_csv('data/train.csv')
dataset.head()

# Converting pandas dataset to numpy arrays
# First one with the 20 features
# Second one with the 'answer'
X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values

# To normalize the 20 features
sc = StandardScaler()
X = sc.fit_transform(X)

# One hot encode the 4 answer classes into 4 binary values
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# Separate the data into training and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

# Creating the NN model, Sequential means the layers are consecutive
model = Sequential()

# Adds layers, relu is a piecewise linear
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Sets loss function as 'categorical_crossentropy'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(X_train, y_train, epochs=100, batch_size=64)

# or this to include validation_data in the data for each epoch
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)

# Gives the model's prediction for the test subsection of the dataset
y_pred = model.predict(X_test)

# Converting predictions to label
# Argmax returns the indices of the maximum values along an axis
# So argmax indicates the class each datapoint most closely aligns with
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
# Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

# Using a built-in function to get the model's accuracy
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

# Using matplotlib to plot model accuracy information
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Using matplotlib to plot model loss information
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()