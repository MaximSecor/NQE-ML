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
    
file_potential = '/Users/maximsecor/Desktop/ANN_1DTISE/TRAINING_DATA/1d_potential.csv'
file_target = '/Users/maximsecor/Desktop/ANN_1DTISE/TRAINING_DATA/1d_energies.csv'
file_wave = '/Users/maximsecor/Desktop/ANN_1DTISE/TRAINING_DATA/1d_wave.csv'

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

std = np.std(target_wave[:,:,0]**2)
mean = np.mean(target_wave[:,:,0]**2)

print((target_wave[:,:,0]**2-mean)/std)

#%%

print((target_wave[:,:,0]**2).shape)
test = np.concatenate((target_wave[:,:,0]**2,target_wave[:,:,1]**2,target_wave[:,:,2]**2,target_wave[:,:,3]**2,target_wave[:,:,4]**2),1)
print(test.shape)

std = np.std(test)
mean = np.mean(test)

test = (test-mean)/std

#%%

print((target_wave**2).shape)

#%%

test_moments = []

xlist = np.linspace(-0.75,0.75,1024)
for i in range(10000):
    test_moments.append(np.array([np.sum(xlist*target_wave[i,:,0]**2),np.sum(xlist**2*target_wave[i,:,0]**2),np.sum(xlist**3*target_wave[i,:,0]**2),np.sum(xlist**4*target_wave[i,:,0]**2),np.sum(xlist**5*target_wave[i,:,0]**2)]))
    
test_moments = np.array(test_moments)

#%%

print(test_moments)


#%%

print(features_potential.shape)

#%%

print("\nConfiguring Model")

dense_layers = 1
dense_nodes = 128

print("Hyper Parameters: ")
print("dense_layers: ", dense_layers)
print("dense_nodes: ", dense_nodes)

train = np.concatenate((target_wave[:,:,0]**2,target_wave[:,:,1]**2,target_wave[:,:,2]**2,target_wave[:,:,3]**2,target_wave[:,:,4]**2),1)
target = features_potential

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 1103)
train_tf = tf.convert_to_tensor(train_X, np.float32)
target_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 1)
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
    model.fit(train_tf, target_tf, epochs=160000, batch_size=16*(2**i), verbose=2, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('\nTraining Time: ',(end-start))

# model.save('saved_model')

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

