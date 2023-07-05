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
target = target_energies*627.50961

print("Loading Data Complete")


#%%

model = load_model(directory_loc + 'ANN_1DTISE/PES_spectra/saved_model')

#%%

predictions_val = model.predict(features)
MAE = mean_absolute_error(target, predictions_val)
print('Cross-Validation Set Error (kcal/mol) =', MAE)

#%%

"Error in energies and excitation energies"

print("Energy Errors")
energy_error = predictions_val - target

print(energy_error)

for i in range(5):
    print("Energy Error of State "+str(int(i))+": ",np.mean(np.abs(energy_error[:,i])))


print("IR errors")

excitations_pred = []
excitations_true = []

for i in range(4):
    pred_IR = predictions_val[:,i+1] - predictions_val[:,0]
    true_IR = target[:,i+1] - target[:,0]
    IR_error = pred_IR - true_IR
    excitations_pred.append(pred_IR)
    excitations_true.append(true_IR)
    print("IR Error of State "+str(int(i+1))+": ",350*np.mean(np.abs(IR_error[i])))
    
excitations_pred = np.array(excitations_pred)
excitations_true = np.array(excitations_true)

#%%

"Error Histogram in wavenumbers"

energy_error_flat = energy_error.reshape(energy_error.shape[0]*energy_error.shape[1])
print(energy_error_flat*350)

ticks = np.linspace(-10,10,20)
plt.hist(energy_error_flat*350, bins=ticks) 
plt.show()

#%%

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_val)))
plot1 = np.zeros((2,len(predictions_val)))
plot2 = np.zeros((2,len(predictions_val)))
plot3 = np.zeros((2,len(predictions_val)))
plot4 = np.zeros((2,len(predictions_val)))


plot4[0] = target[:,4]
plot4[1] = predictions_val[:,4]
data_values = pd.DataFrame(plot4.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plot3[0] = target[:,3]
plot3[1] = predictions_val[:,3]
data_values = pd.DataFrame(plot3.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
plot2[0] = target[:,2]
plot2[1] = predictions_val[:,2]
data_values = pd.DataFrame(plot2.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plot1[0] = target[:,1]
plot1[1] = predictions_val[:,1]
data_values = pd.DataFrame(plot1.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
plot0[0] = target[:,0]
plot0[1] = predictions_val[:,0]
data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')

plt.savefig(directory_loc + 'ANN_1DTISE/PES_spectra/Pot_energies.png',dpi=1200)

#%%

"Excitation Energy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(excitations_pred[0])))
plot1 = np.zeros((2,len(excitations_pred[0])))
plot2 = np.zeros((2,len(excitations_pred[0])))
plot3 = np.zeros((2,len(excitations_pred[0])))

plot3[0] = excitations_true[3]
plot3[1] = excitations_pred[3]
data_values = pd.DataFrame(plot3.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
plot2[0] = excitations_true[2]
plot2[1] = excitations_pred[2]
data_values = pd.DataFrame(plot2.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plot1[0] = excitations_true[1]
plot1[1] = excitations_pred[1]
data_values = pd.DataFrame(plot1.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
plot0[0] = excitations_true[0]
plot0[1] = excitations_pred[0]
data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')

plt.savefig(directory_loc + 'ANN_1DTISE/PES_spectra/Pot_excitations.png',dpi=1200)











