#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:03:37 2020

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

n_sample = len(features_potential)
target_wave = target_wave.reshape(n_sample,1024,5)
domain = np.linspace(-0.5,1.5,1024)

features_potential[features_potential>0.159] = 0.159

features = features_potential
target_0 = target_wave[:,:,0]
target_1 = target_wave[:,:,1]
target_2 = target_wave[:,:,2]
target_3 = target_wave[:,:,3]
target_4 = target_wave[:,:,4]

print("Loading Data Complete")

#%%

def custom_loss(y_true, y_pred):

    overlap = K.sum(y_true*y_pred,1)
    loss_1 = K.sum(K.square(1-K.square(overlap)))

    self_overlap = K.sum(y_pred*y_pred,1)
    loss_2 = K.sum(K.square(1-self_overlap))

    loss = loss_1 + loss_2

    return loss

#%%

model_0 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_0/saved_model', custom_objects={'custom_loss':custom_loss})
model_1 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_1/saved_model', custom_objects={'custom_loss':custom_loss})
# model_2 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_2/saved_model', custom_objects={'custom_loss':custom_loss})
# model_3 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_3/saved_model', custom_objects={'custom_loss':custom_loss})
# model_4 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_4/saved_model', custom_objects={'custom_loss':custom_loss})

#%%

"Error in density"

predictions_0 = model_0.predict(features)
MAE = mean_absolute_error(target_0**2, predictions_0**2)
print('Cross-Validation Set Error (kcal/mol) =', MAE)

predictions_1 = model_1.predict(features)
MAE = mean_absolute_error(target_1**2, predictions_1**2)
print('Cross-Validation Set Error (kcal/mol) =', MAE)

#%%

for i in range(10):
    q = np.random.randint(0,10000)
    print(q)
    plt.plot(domain,predictions_0_norm[q],domain,features[q])
    plt.show()

#%%

norm_fact = np.sqrt(1/np.sum(predictions_0**2,1))
predictions_0_norm = (predictions_0.T * norm_fact).T

print("Bad Wavefunctions")
print(np.where(np.abs(1-np.abs(np.sum(target_0*predictions_0_norm,1)))>0.01))
print("Excellent Wavefunctions")
print(np.where(np.abs(1-np.abs(np.sum(target_0*predictions_0_norm,1)))<0.00001))

#%%

i = 1266
plt.plot(domain,predictions_0_norm[i],domain,target_0[i])
plt.show()

#%%

plt.plot(domain,predictions_0_norm[i],domain,features[i])
plt.show()


#%%

c = 1

if np.sum(target_0[i]*predictions_0_norm[i]) < 0:
    c = -1

N = 24
grid_size = len(domain)
start = (int(N/2)-1)
finish = grid_size-(int(N/2))
predictions_val_smooth = c*running_mean(predictions_0[i],N)

norm_fact = np.sqrt(1/np.sum(predictions_val_smooth**2))
predictions_val_smooth = predictions_val_smooth*norm_fact


a = 300
b = 40

plt.ylim(-1,99)
plt.plot(domain,627*features[i],domain[start:finish], a*target_0[i,start:finish]+b,domain[start:finish],a*predictions_val_smooth+b)
plt.show

# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/Pot_wave_2a.png',dpi=1200)

#%%

norm_fact = np.sqrt(1/np.sum(predictions_0**2,1))
predictions_0_norm = (predictions_0.T * norm_fact).T

true_pos = np.sum(domain*target_0**2,1)
pred_pos = np.sum(domain*predictions_0_norm**2,1)
error_pos = true_pos - pred_pos

plot_pos = np.zeros((2,len(predictions_val)))

plot_pos[0] = true_pos
plot_pos[1] = pred_pos
data_values = pd.DataFrame(plot_pos.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plt.ylim(-0.1,1.0)
plt.xlim(-0.1,1.0)

# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/Pot_wave_position.png',dpi=1200)


#%%

ticks = np.linspace(-0.01,0.01,20)
plt.hist(error_pos, bins=ticks) 
plt.show()

#%%

overlap_error = np.abs(np.sum(predictions_0_norm*target_0,1))
ticks = np.linspace(0.999,1.0,40)
plt.hist(overlap_error, bins=ticks)
plt.show()










