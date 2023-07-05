#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:01:34 2020

@author: maximsecor
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as time
import os
import sys
import seaborn as sns; sns.set()

from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
import keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 

from scipy.signal import argrelextrema

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

np.set_printoptions(precision=4,suppress=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

#%%

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#%%

directory_loc = '/Users/maximsecor/Desktop/'

print("\n----------------Welcome to QMML----------------")
print("\nLoading Data")
    
file_potential = directory_loc + 'ANN_1DTISE/TRAINING_DATA/1d_potential.csv'
file_target = directory_loc + 'ANN_1DTISE/TRAINING_DATA/1d_energies.csv'
file_wave = directory_loc + 'ANN_1DTISE/TRAINING_DATA/1d_wave.csv'

data_pot = pd.read_csv(file_potential)
data_target = pd.read_csv(file_target)
data_wave = pd.read_csv(file_wave)

features_potential = data_pot.values
target_energies = data_target.values
target_wave = data_wave.values

features = target_energies*627.50961
features_potential[features_potential>0.159] = 0.159
target = features_potential

domain = np.linspace(-0.5,1.5,1024)

print("Loading Data Complete")

#%%

model = load_model(directory_loc + 'ANN_1DTISE/spectra_PES/saved_model')

#%%

predictions_val = model.predict(features)
MAE = mean_absolute_error(target, predictions_val)
print('Cross-Validation Set Error =', MAE)

#%%

error = np.abs(np.sum(predictions_val-target,1))
print(np.where(error>0.5))
print(np.where(error<0.05))

#%%

temp = np.where(error<0.0005)
print(temp[0])

#%%

temp = np.where(error>5)
print(temp[0])

#%%

for i in temp[0]:
    print(i)
    plt.plot(domain,predictions_val[i],domain,target[i])
    plt.show()
    
#%%

i = temp[0][20]
plt.plot(domain,predictions_val[i],domain,target[i])
plt.show()
    
#%%

N = 24
grid_size = len(domain)
start = (int(N/2)-1)
finish = grid_size-(int(N/2))
predictions_val_smooth = running_mean(predictions_val[i],N)

a = 627
b = 0

plt.ylim(-5,90)
plt.plot(domain[start:finish],a*target[i,start:finish],domain[start:finish],a*predictions_val_smooth)
plt.show
# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/Den_Pot.png',dpi=1200)

#%%

q = 10000

N = 24
predictions_val_smooth = []

for i in range(q):
    predictions_val_smooth.append(running_mean(predictions_val[i], N))
    
predictions_val_smooth = np.array(predictions_val_smooth)
predictions_val_smooth[predictions_val_smooth>(50/627)] = (50/627)

#%%

plt.plot(predictions_val_smooth[i])
plt.show()

#%%

plt.plot(target[i])
plt.show()

#%%


n=10
n2=50

rel_error = []
bar_error = []
pos_error = []

for i in range(q):
    
    n_minima = []
    rel_en = []
    rel_pos = []
    bar1 = []
    bar2 = []
    
    "Potential"
    a = predictions_val_smooth[i]
    
    "Minima"
    maxInd = argrelextrema(a, np.less, order=n)
    r = a[maxInd]
    if len(r)>1:
        n_minima.append(2)
        rel_en.append(r[0]-r[1])
        pos = np.array(maxInd)*(2/1024)-1
        rel_pos.append(pos[0,1]-pos[0,0])
    else:
        n_minima.append(1)
        
    "Barriers"
    maxIndbar = argrelextrema(a, np.greater, order = n2)
    rbar = a[maxIndbar]
    if len(rbar)==1 and len(r)>1:
        if r[0] > r[1]:
            bar1.append(rbar-r[1])
            bar2.append(rbar-r[0])
        else:
            bar1.append(rbar-r[0])
            bar2.append(rbar-r[1])
            
    if len(r)>1 and len(rbar)==1:
        
        n_minima_true = []
        rel_en_true = []
        rel_pos_true = []
        bar1_true = []
        bar2_true = []        
    
        "Potential"
        a = target[i]
        
        "Minima"
        maxInd = argrelextrema(a, np.less, order=n)
        r = a[maxInd]
        if len(r)>1:
            n_minima_true.append(2)
            rel_en_true.append(r[0]-r[1])
            pos = np.array(maxInd)*(2/1024)-1
            rel_pos_true.append(pos[0,1]-pos[0,0])
        else:
            n_minima_true.append(1)
            
        "Barriers"
        maxIndbar = argrelextrema(a, np.greater)
        rbar = a[maxIndbar]
        if len(rbar)==1 and len(r)>1:
            if r[0] > r[1]:
                bar1_true.append(rbar-r[1])
                bar2_true.append(rbar-r[0])
            else:
                bar1_true.append(rbar-r[0])
                bar2_true.append(rbar-r[1])
                
        if len(r)>1 and len(rbar)==1:
        
            print(i)
            rel_error.append((rel_en[0]-rel_en_true[0])*627)
            bar_error.append((bar1[0][0]-bar1_true[0][0])*627)
            pos_error.append(rel_pos[0]-rel_pos_true[0])
            

rel_error = np.array(rel_error)
bar_error = np.array(bar_error)
pos_error = np.array(pos_error)
            
#%%

print(np.mean(np.abs(rel_error)))
print(np.mean(np.abs(bar_error)))
print(np.mean(np.abs(pos_error)))

#%%

test = np.histogram(rel_error)
print(test[0])
ticks = np.linspace(-2.5,2.5,20)
plt.hist(rel_error, bins=ticks)
# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/potentials_rel_en.png',dpi=1200)
plt.show()

#%%

test = np.histogram(bar_error)
print(test[0])
ticks = np.linspace(-2.5,2.5,20)
plt.hist(bar_error, bins=ticks)
# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/potentials_rel_en.png',dpi=1200)
plt.show()

#%%

test = np.histogram(pos_error)
print(test[0])
ticks = np.linspace(-0.1,0.1,20)
plt.hist(pos_error, bins=ticks)
# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/potentials_rel_en.png',dpi=1200)
plt.show()









