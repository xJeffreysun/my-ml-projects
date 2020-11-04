#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:06:21 2020

@author: jeffreysun
"""

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import keras

# preprocessing

dataset = pd.read_csv("/Users/jeffreysun/Desktop/my-ml-projects/temperature/temp_params.csv")
dataset.head()
X = dataset.iloc[:,:6].values
y = dataset.iloc[:,6:7].values
cd 
# normalization

sc = StandardScaler()
X = sc.fit_transform(X)

# splitting testing and training data

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.4)

# building the model

model = Sequential()
model.add(Dense(100, input_dim=6, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))
model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

# running it

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=40)

# plotting with matplotlib

plt.plot(history.history['val_mean_squared_error'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()