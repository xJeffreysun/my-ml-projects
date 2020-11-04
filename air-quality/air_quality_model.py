#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:59:27 2020

@author: jeffreysun
"""

# import dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# import dataset

data = pd.read_excel("/Users/jeffreysun/Desktop/my-ml-projects/air-quality/AirQualityDatasets/AirQualityUCI.xlsx")

# -200 stands for na data
# data cleaning to drop na data

data = data.replace(-200, np.nan)
data = data.dropna()
data.describe()

# columns 2 to 12 are the features
# column 13 is the output

X = data.iloc[:, 2:13]
Y = data.iloc[:, 13:14]

# normalizating the features

sc = StandardScaler()
X = sc.fit_transform(X)
Y = sc.fit_transform(Y)

# splitting to train and test data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1)

# creating a new model

model = Sequential()

model.add(Dense(100, input_dim=11, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(1))

model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

history = model.fit(X_train, Y_train,validation_data = (X_test,Y_test), epochs=300, batch_size=64)


plt.plot(history.history['val_mean_squared_error'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()