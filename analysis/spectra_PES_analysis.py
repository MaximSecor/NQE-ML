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
# import seaborn as sns; sns.set()
import seaborn as sns

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

n_sample = len(features_potential)
target_wave = target_wave.reshape(n_sample,1024,5)

features_potential[features_potential>0.159] = 0.159
features = target_energies*627
target = features_potential

domain = np.linspace(-0.5,1.5,1024)

print("Loading Data Complete")

#%%

train_X, test_X, train_y, test_y = train_test_split(features, target, test_size = 0.1, random_state = 1103)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.11111, random_state = 120)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

#%%

model = load_model(directory_loc + 'ANN_1DTISE/spectra_PES/saved_model')
predictions_test = model.predict(test_X_tf)

#%%

temp_2 = np.sum(np.abs(predictions_test-test_y),1)
temp_2 = np.array(temp_2*627)

print(temp_2)

best_1 = np.where((temp_2)<300)[0]
best_2 = np.where((temp_2)>0)[0]
best = np.intersect1d(best_1,best_2)
print(best)


#%%

for q in range(len(best)):

    i = best[q]
    
    N = 24
    grid_size = len(domain)
    start = (int(N/2)-1)
    finish = grid_size-(int(N/2))
    
    test_pred_smooth = running_mean(predictions_test[i],N)
    
    add = test_X[i]
    scale = add[4]*1.2
    scale_wave = scale*0.75
    
    plt.ylim(-1,scale)
    
    # plt.ylim(-1,50)
    # plt.plot(domain,val_X[i]*627,'k')
    plt.plot(domain,test_y[i]*627,'r')
    plt.plot(domain[start:finish],test_pred_smooth*627,'--b')
    
    plt.xlabel('Position (Å)')
    plt.ylabel('Energy (kcal/mol)')
    
    plt.rc('font', family='Helvetica')
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # plt.hlines(add[0], -0.55, -0.3, colors='k', linewidth=2)
    # plt.hlines(add[1], -0.55, -0.3, colors='k', linewidth=2)
    # plt.hlines(add[2], -0.55, -0.3, colors='k', linewidth=2)
    # plt.hlines(add[3], -0.55, -0.3, colors='k', linewidth=2)
    # plt.hlines(add[4], -0.55, -0.3, colors='k', linewidth=2)
    
    plt.hlines(add[0], -0.55, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[1], -0.55, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[2], -0.55, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[3], -0.55, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[4], -0.55, 1.5, colors='k', linewidth=0.5)
    
    plt.text(-0.5, add[0]-1.5, r'$E_{0}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[1]+0.5, r'$E_{1}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[2]+0.5, r'$E_{2}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[3]+0.5, r'$E_{3}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[4]+0.5, r'$E_{4}$', fontsize=SMALL_SIZE)
    
    plt.show()
    # plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/spectra_PES_ADW.tif',dpi=1200)

#%%

i = 866

N = 24
grid_size = len(domain)
start = (int(N/2)-1)
finish = grid_size-(int(N/2))

test_pred_smooth = running_mean(predictions_test[i],N)

add = test_X[i]
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)

# plt.ylim(-1,50)
# plt.plot(domain,val_X[i]*627,'k')
plt.plot(domain,test_y[i]*627,'r')
plt.plot(domain[start:finish],test_pred_smooth*627,'--b')

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')

plt.rc('font', family='Helvetica')

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.hlines(add[0], -0.55, -0.3, colors='k', linewidth=2)
# plt.hlines(add[1], -0.55, -0.3, colors='k', linewidth=2)
# plt.hlines(add[2], -0.55, -0.3, colors='k', linewidth=2)
# plt.hlines(add[3], -0.55, -0.3, colors='k', linewidth=2)
# plt.hlines(add[4], -0.55, -0.3, colors='k', linewidth=2)

plt.hlines(add[0], -0.55, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.55, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.55, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.55, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.55, 1.5, colors='k', linewidth=0.5)

plt.text(-0.5, add[0]-1.5, r'$E_{0}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[1]+0.5, r'$E_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.5, r'$E_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$E_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$E_{4}$', fontsize=SMALL_SIZE)

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/spectra_PES_SWS.tif',dpi=1200)


#%%

predictions_train = model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

predictions_val = model.predict(val_X_tf)
MAE = mean_absolute_error(val_y_tf, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error =', MAE)

#%%

error = np.sum(np.abs(predictions_test-test_y),1)
print(np.where(error>5))
print(np.where(error<5))

#%%

print(len(np.where(error>25)[0]))
print(len(np.where(error<0.5)[0]))

#%%

temp = np.where(error<5)
# temp = np.where(error>25)
print(temp[0][0:5])
print(len(temp[0]))
good_sample = temp[0][0:10]

#%%

for i in good_sample:
    print(i)
    plt.ylim(-1,50)
    plt.plot(domain,predictions_test[i]*627,domain,test_y[i]*627)
    plt.show()
    
#%%

i = temp[0][20]
plt.plot(domain,predictions_test[i],domain,test_y[i])
plt.show()
    
#%%

for i in good_sample:
    N = 24
    grid_size = len(domain)
    start = (int(N/2)-1)
    finish = grid_size-(int(N/2))
    predictions_val_smooth = running_mean(predictions_test[i],N)
    
    a = 627
    b = 0
    
    plt.ylim(-5,90)
    plt.plot(domain[start:finish],a*test_y[i,start:finish],domain[start:finish],a*predictions_val_smooth)
    plt.show()
    # plt.savefig('/Users/maximsecor/Desktop/Models_ANN/Den_Pot.png',dpi=1200)

#%%

q = 1000

N = 24
predictions_val_smooth = []

for i in range(q):
    predictions_val_smooth.append(running_mean(predictions_test[i], N))
    
predictions_val_smooth = np.array(predictions_val_smooth)
predictions_val_smooth[predictions_val_smooth>(90/627)] = (90/627)

#%%

plt.plot(predictions_val_smooth[i])
plt.show()

#%%

plt.plot(test_y[i])
plt.show()

#%%

q = 1000

val = 50

u50_error_tracker = []
for i in range(q):
    predictions_val_smooth_u50_idx = np.where(predictions_val_smooth[i]<val/627)[0]
    predictions_val_smooth_u50 = predictions_val_smooth[i,predictions_val_smooth_u50_idx]
    val_y_u50 = test_y[i,predictions_val_smooth_u50_idx]
    u50_error_tracker.append(np.mean(predictions_val_smooth_u50-val_y_u50)*627)

u50_error_tracker = np.array(u50_error_tracker)
print(np.mean(np.abs(u50_error_tracker)))

#%%

print(u50_error_tracker)

#%%

temp_2 = np.array(np.abs(u50_error_tracker))

print(temp_2)

best_1 = np.where((temp_2)<0.05)[0]
best_2 = np.where((temp_2)>0)[0]
best = np.intersect1d(best_1,best_2)
print(best)

#%%

for q in range(len(best)):

    i = best[q]
    
    N = 24
    grid_size = len(domain)
    start = (int(N/2)-1)
    finish = grid_size-(int(N/2))
    
    test_pred_smooth = running_mean(predictions_test[i],N)
    
    add = test_X[i]
    scale = add[4]*1.2
    scale_wave = scale*0.75
    
    plt.ylim(-1,scale)
    
    # plt.ylim(-1,50)
    # plt.plot(domain,val_X[i]*627,'k')
    plt.plot(domain,test_y[i]*627,'r')
    plt.plot(domain[start:finish],test_pred_smooth*627,'--b')
    
    plt.xlabel('Position (Å)')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('Example: ' + str(i))
    
    plt.rc('font', family='Helvetica')
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # plt.hlines(add[0], -0.55, -0.3, colors='k', linewidth=2)
    # plt.hlines(add[1], -0.55, -0.3, colors='k', linewidth=2)
    # plt.hlines(add[2], -0.55, -0.3, colors='k', linewidth=2)
    # plt.hlines(add[3], -0.55, -0.3, colors='k', linewidth=2)
    # plt.hlines(add[4], -0.55, -0.3, colors='k', linewidth=2)
    
    plt.hlines(add[0], -0.55, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[1], -0.55, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[2], -0.55, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[3], -0.55, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[4], -0.55, 1.5, colors='k', linewidth=0.5)
    
    plt.text(-0.5, add[0]-1.5, r'$E_{0}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[1]+0.5, r'$E_{1}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[2]+0.5, r'$E_{2}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[3]+0.5, r'$E_{3}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[4]+0.5, r'$E_{4}$', fontsize=SMALL_SIZE)
    
    plt.show()
    # plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/spectra_PES_ADW.tif',dpi=1200)

#%%

i = 65

N = 24
grid_size = len(domain)
start = (int(N/2)-1)
finish = grid_size-(int(N/2))

test_pred_smooth = running_mean(predictions_test[i],N)

add = test_X[i]
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)

# plt.ylim(-1,50)
# plt.plot(domain,val_X[i]*627,'k')
plt.plot(domain,test_y[i]*627,'r')
plt.plot(domain[start:finish],test_pred_smooth*627,'--b')

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
# plt.title('Example: ' + str(i))

plt.rc('font', family='Helvetica')

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.hlines(add[0], -0.55, -0.3, colors='k', linewidth=2)
# plt.hlines(add[1], -0.55, -0.3, colors='k', linewidth=2)
# plt.hlines(add[2], -0.55, -0.3, colors='k', linewidth=2)
# plt.hlines(add[3], -0.55, -0.3, colors='k', linewidth=2)
# plt.hlines(add[4], -0.55, -0.3, colors='k', linewidth=2)

plt.hlines(add[0], -0.55, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.55, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.55, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.55, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.55, 1.5, colors='k', linewidth=0.5)

plt.text(-0.5, add[0]+0.5, r'$E_{0}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[1]+0.5, r'$E_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.5, r'$E_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$E_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$E_{4}$', fontsize=SMALL_SIZE)

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/spectra_PES_ASW.tif',dpi=1200)

#%%

i = 1

plt.ylim(-5,90)
plt.plot(domain[start:finish],a*test_y[i,start:finish],domain[start:finish],a*predictions_val_smooth[i])
plt.show

domain_pred = domain[start:finish]

#%%

i = 1

failures_1 = 0
failures_2 = 0

SW_freq_error = []

DW_min_mag_err = []
DW_min_pos_err = []
DW_max_mag_err = []
DW_max_pos_err = []

n = 32

for i in range(1000):
    
    print("\n")
    
    maxInd = argrelextrema(test_y[i], np.less, order=n)
    true_minima = test_y[i,maxInd[0]]
    true_minima_pos = domain[maxInd[0]]
    
    if len(true_minima_pos) == 1:
        print(i," -> 1 minima")
        n_min = 1
    else: 
        print(i," -> 2 minima")
        n_min = 2
        
    if n_min == 1:
            
        maxInd = argrelextrema(predictions_val_smooth[i], np.less, order=n)
        pred_minima = predictions_val_smooth[i,maxInd[0]]
        pred_minima_pos = domain_pred[maxInd[0]]
        
        if len(pred_minima_pos) == 1:
            
            print("predicited PES also has 1 minima")
            
            dd_true_y = np.gradient(np.gradient(test_y[i]))/(((2*1.89)/1024)**2)
            dd_pred_y = np.gradient(np.gradient(predictions_val_smooth[i]))/(((2*1.89)/1024)**2)
            
            print("First transition frequency: ", test_X[i,1]-test_X[i,0])
            print("Finite Difference True: ", np.sqrt(dd_true_y[maxInd[0]]/1836)*627)
            print("Finite Difference NN: ", np.sqrt(dd_pred_y[maxInd[0]]/1836)*627)
            print("Error: ", (np.sqrt(dd_true_y[maxInd[0]]/1836)*627) - np.sqrt(dd_pred_y[maxInd[0]]/1836)*627)
            
            SW_freq_error.append((np.sqrt(dd_true_y[maxInd[0]]/1836)*627) - np.sqrt(dd_pred_y[maxInd[0]]/1836)*627)
            
        else:

            print("NN failed to predict PES with 1 minima")
            failures_1 = failures_1 + 1
            
    if n_min == 2:
            
        maxInd = argrelextrema(predictions_val_smooth[i], np.less, order=n)
        pred_minima = predictions_val_smooth[i,maxInd[0]]
        pred_minima_pos = domain_pred[maxInd[0]]
        
        if len(pred_minima_pos) == 2:
            
            print("NN predicited PES also has 2 minima")
            
            print("Second Minima Position: ", true_minima_pos[1], pred_minima_pos[1], true_minima_pos[1] - pred_minima_pos[1])
            print("Second Minima Height: ", true_minima[1], pred_minima[1], true_minima[1] - pred_minima[1])
            
            
            maxInd = argrelextrema(test_y[i], np.greater, order=n)
            true_maxima = test_y[i,maxInd[0]]
            true_maxima_pos = domain[maxInd[0]]
            
            maxInd = argrelextrema(predictions_val_smooth[i], np.greater, order=n)
            pred_maxima = predictions_val_smooth[i,maxInd[0]]
            pred_maxima_pos = domain_pred[maxInd[0]]
            
            DW_min_mag_err.append(true_minima[1] - pred_minima[1])
            DW_min_pos_err.append(true_minima_pos[1] - pred_minima_pos[1])

            if len(true_maxima) == 1 and len(pred_maxima) == 1:
            
                print("TST Position: ", true_maxima_pos[0], pred_maxima_pos[0], true_maxima_pos[0] - pred_maxima_pos[0])
                print("TST Height: ", true_maxima[0], pred_maxima[0], true_maxima[0] - pred_maxima[0])
                
                DW_max_mag_err.append(true_maxima[0] - pred_maxima[0])
                DW_max_pos_err.append(true_maxima_pos[0] - pred_maxima_pos[0])
            
        else:

            print("NN failed to predict PES with 2 minima")
            failures_2 = failures_2 + 1
            
print(failures_1,failures_2)

#%%

print(np.mean(np.abs(SW_freq_error))*350)

print(np.mean(np.abs(DW_min_pos_err)))
print(np.mean(np.abs(DW_min_mag_err))*627)
print(np.mean(np.abs(DW_max_pos_err)))
print(np.mean(np.abs(DW_max_mag_err))*627)

SW_freq_error = (np.array(SW_freq_error).reshape(len(SW_freq_error)))
DW_min_mag_err = (np.array(DW_min_mag_err).reshape(len(DW_min_mag_err)))
DW_max_mag_err = (np.array(DW_max_mag_err).reshape(len(DW_max_mag_err)))

#%%

ticks = np.linspace(-1500,1500,50)
plt.hist(SW_freq_error*350, bins=ticks)
plt.show()

#%%

ticks = np.linspace(-0.2,0.2,50)
plt.hist(DW_min_pos_err, bins=ticks)
plt.show()

#%%

ticks = np.linspace(-5,5,50)
plt.hist(DW_min_mag_err*627, bins=ticks)
plt.show()

#%%

ticks = np.linspace(-0.2,0.2,50)
plt.hist(DW_max_pos_err, bins=ticks)
plt.show()

#%%

ticks = np.linspace(-5,5,50)
plt.hist(DW_max_mag_err*627, bins=ticks)
plt.show()








