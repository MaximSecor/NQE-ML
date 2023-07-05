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
# import seaborn as sns; sns.set()
import seaborn as sns

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

features = features_potential
target = target_energies*627.50961

train_X, test_X, train_y, test_y = train_test_split(features, target, test_size = 0.1, random_state = 1103)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.11111, random_state = 120)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

#%%

print(np.max(train_y[:,1]-train_y[:,0])*350)
print(np.min(train_y[:,1]-train_y[:,0])*350)

print(np.max(test_y[:,1]-test_y[:,0])*350)
print(np.min(test_y[:,1]-test_y[:,0])*350)

print(np.max(train_y[:,1]-train_y[:,0])*1)
print(np.min(train_y[:,1]-train_y[:,0])*1)

print(np.max(test_y[:,1]-test_y[:,0])*1)
print(np.min(test_y[:,1]-test_y[:,0])*1)

#%%

model = load_model(directory_loc + 'ANN_1DTISE/PES_spectra/saved_model')

#%%

predictions_train = model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error (kcal/mol) =', MAE)

predictions_val = model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error (kcal/mol) =', MAE)

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error (kcal/mol) =', MAE)

#%%

"Error in energies and excitation energies"

print("Energy Errors")
energy_error = predictions_test - test_y

print(np.mean(abs(energy_error)))

for i in range(5):
    print("Energy Error of State "+str(int(i))+": ",np.mean(np.abs(energy_error[:,i])))


print("IR errors")

excitations_pred = []
excitations_true = []

for i in range(4):
    pred_IR = predictions_test[:,i+1] - predictions_test[:,0]
    true_IR = test_y[:,i+1] - test_y[:,0]
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

ticks = np.linspace(-25,25,40)
plt.hist(energy_error_flat*350, bins=ticks) 
plt.show()

#%%

"Potentials with greatest error"

print(np.argmax(energy_error[:,0]))
print(np.argmax(energy_error[:,1]))
print(np.argmax(energy_error[:,2]))
print(np.argmax(energy_error[:,3]))
print(np.argmax(energy_error[:,4]))

plt.ylim(-1,50)
plt.plot(domain,val_X[np.argmax(energy_error[:,0])]*627)
plt.show()

plt.ylim(-1,50)
plt.plot(domain,val_X[np.argmax(energy_error[:,1])]*627)
plt.show()

plt.ylim(-1,50)
plt.plot(domain,val_X[np.argmax(energy_error[:,2])]*627)
plt.show()

plt.ylim(-1,50)
plt.plot(domain,val_X[np.argmax(energy_error[:,3])]*627)
plt.show()

plt.ylim(-1,50)
plt.plot(domain,val_X[np.argmax(energy_error[:,4])]*627)
plt.show()

#%%

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_test)))
plot1 = np.zeros((2,len(predictions_test)))
plot2 = np.zeros((2,len(predictions_test)))
plot3 = np.zeros((2,len(predictions_test)))
plot4 = np.zeros((2,len(predictions_test)))

plot4[0] = test_y_tf[:,4]
plot4[1] = predictions_test[:,4]
data_values = pd.DataFrame(plot4.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plot3[0] = test_y_tf[:,3]
plot3[1] = predictions_test[:,3]
data_values = pd.DataFrame(plot3.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plot2[0] = test_y_tf[:,2]
plot2[1] = predictions_test[:,2]
data_values = pd.DataFrame(plot2.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plot1[0] = test_y_tf[:,1]
plot1[1] = predictions_test[:,1]
data_values = pd.DataFrame(plot1.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
plot0[0] = test_y_tf[:,0]
plot0[1] = predictions_test[:,0]
data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
# ax.set_ylabel('ANN Energy (kcal/mol)')
# ax.set_xlabel('FGH Energy (kcal/mol)')
# ax.set_title('Energy Comaprison')

plt.rc('font', family='Helvetica')

plt.axes().set_aspect('equal')
plt.rcParams["figure.figsize"] = (4,4)

plt.xticks(np.arange(0, 60, step=10))
plt.yticks(np.arange(0, 60, step=10))

x = np.linspace(0,np.max(test_y_tf[:,4])+1,1000)
y = x
plt.plot(x, y, '--k', linewidth=1.5)

SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_energies.tiff',dpi=1200)

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

ax.set_ylabel('ANN Excitation Energy Prediction (kcal/mol)')
ax.set_xlabel('FGH Excitation Energy (kcal/mol)')
ax.set_title('Excitation Energy Comaprison')

plt.rc('font', family='Helvetica')
plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_excitations.tiff',dpi=1200)











