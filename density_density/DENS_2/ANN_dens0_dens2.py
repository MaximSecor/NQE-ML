#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:36:38 2020

@author: maximsecor
"""

import pandas as pd
import numpy as np
import time as time
import os
import sys

from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

np.set_printoptions(precision=4,suppress=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

#%%

print("\n----------------Welcome to QMML----------------")
print("\nLoading Data")
    
file_potential = '../../training_data/1d_potential.csv'
file_target = '../../training_data/1d_energies.csv'
file_wave = '../../training_data/1d_wave.csv'

data_pot = pd.read_csv(file_potential)
data_target = pd.read_csv(file_target)
data_wave = pd.read_csv(file_wave)

features_potential = data_pot.values
target_energies = data_target.values
target_wave = data_wave.values

n_sample = len(features_potential)
target_wave = target_wave.reshape(n_sample,1024,5)
domain = np.linspace(-0.5,1.5,1024)
features_potential[features_potential>0.159] = 0.159

print("Loading Data Complete")

#%%

print("\nConfiguring Model")

dense_layers = 2
dense_nodes = 768

print("Hyper Parameters: ")
print("dense_layers: ", dense_layers)
print("dense_nodes: ", dense_nodes)

train = target_wave[:,:,0]**2 
target = target_wave[:,:,2]**2 

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 1103)
train_tf = tf.convert_to_tensor(train_X, np.float32)
target_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 1000)
model = Sequential()
model.add(Dense(dense_nodes, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(dense_layers):
    model.add(Dense(dense_nodes, kernel_initializer='normal',activation='relu'))
model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
opt = Adam(learning_rate=(0.00003))
model.compile(loss='mean_squared_error', optimizer=opt)

print("\nModel Configured")
start = time.time()
for i in range(9):
    model.fit(train_tf, target_tf, epochs=160000, batch_size=16*(2**i), verbose=0, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('\nTraining Time: ',(end-start))

model.save('saved_model')

#%%

print("\nResults")
start = time.time()

predictions_val = model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = model.predict(train_tf)
MAE = mean_absolute_error(target_tf, predictions_train)
print('Training Set Error =', MAE)

end = time.time()
print('\nPrediction Time: ',(end-start)/len(features_potential))




