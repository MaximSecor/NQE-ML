#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:04:26 2020

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
features = target_wave[:,:,0]**2
features = features/np.max(features)


features_potential[features_potential>0.159] = 0.159
target = features_potential

domain = np.linspace(-0.5,1.5,1024)

print("Loading Data Complete")

#%%

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size = 0.1, random_state = 1103)
train_tf = tf.convert_to_tensor(train_X, np.float32)
target_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

train_X, val_X, etrain_y, ener_y = train_test_split(features, target_energies, test_size = 0.1, random_state = 1103)

#%%

model = load_model(directory_loc + 'ANN_1DTISE/minmax_PES/saved_model')
predictions_val = model.predict(val_X_tf)

#%%

q = 1000
filler = np.full(23,100/627)

N = 24
predictions_val_smooth = []

for i in range(q):
    temp = running_mean(predictions_val[i], N)
    temp = np.concatenate((filler[:11],temp))
    temp = np.concatenate((temp,filler[11:]))
    predictions_val_smooth.append(temp)
    
predictions_val_smooth = np.array(predictions_val_smooth)
    
#%%

i = 7
add = ener_y[i]*627
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)
plt.plot(domain,val_X[i]*1000+add[0],'k')
plt.plot(domain,val_y[i]*627,'r')
plt.plot(domain,predictions_val_smooth[i]*627,'--b')

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')

plt.rc('font', family='Helvetica')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.text(-0.5, add[0]+0.5, r'$\rho_{0}$', fontsize=SMALL_SIZE)

# plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/den_PES_ADW.tif',dpi=1200)

#%%

i = 50
add = ener_y[i]*627
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)
plt.plot(domain,val_X[i]*1000+add[0],'k')
plt.plot(domain,val_y[i]*627,'r')
plt.plot(domain,predictions_val_smooth[i]*627,'--b')

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')

plt.rc('font', family='Helvetica')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.text(-0.5, add[0]+0.5, r'$\rho_{0}$', fontsize=SMALL_SIZE)

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images/den_PES_HO.tif',dpi=1200)

#%%

i = 934
add = ener_y[i]*627
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)
plt.plot(domain,val_X[i]*1000+add[0],'k')
plt.plot(domain,val_y[i]*627,'r')
plt.plot(domain,predictions_val_smooth[i]*627,'--b')

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')

plt.rc('font', family='Helvetica')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.text(-0.5, add[0]+0.5, r'$\rho_{0}$', fontsize=SMALL_SIZE)

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images/den_PES_SDW.tif',dpi=1200)

#%%

i = 66
add = ener_y[i]*627
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)
plt.plot(domain,val_X[i]*1000+add[0],'k')
plt.plot(domain,val_y[i]*627,'r')
plt.plot(domain,predictions_val_smooth[i]*627,'--b')

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')

plt.rc('font', family='Helvetica')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.text(-0.5, add[0]+0.5, r'$\rho_{0}$', fontsize=SMALL_SIZE)

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images/den_PES_SWS.tif',dpi=1200)

#%%

predictions_val = model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = model.predict(train_tf)
MAE = mean_absolute_error(target_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

q = 1000

N = 24
predictions_val_smooth = []

for i in range(q):
    predictions_val_smooth.append(running_mean(predictions_val[i], N))
    
predictions_val_smooth = np.array(predictions_val_smooth)
predictions_val_smooth[predictions_val_smooth>(90/627)] = (90/627)

#%%

q = 1000

u50_error_tracker = []
for i in range(q):
    predictions_val_smooth_u50_idx = np.where(predictions_val_smooth[i]<50/627)[0]
    predictions_val_smooth_u50 = predictions_val_smooth[i,predictions_val_smooth_u50_idx]
    val_y_u50 = val_y[i,predictions_val_smooth_u50_idx]
    u50_error_tracker.append(np.mean(predictions_val_smooth_u50-val_y_u50)*627)

u50_error_tracker = np.array(u50_error_tracker)
print(np.mean(np.abs(u50_error_tracker)))

#%%

plt.plot(predictions_val_smooth[i])
plt.show()

#%%

plt.plot(val_y[i])
plt.show()

#%%

i = 20

plt.ylim(-5,90)
plt.plot(domain[start:finish],a*val_y[i,start:finish],domain[start:finish],a*predictions_val_smooth[i])
plt.show

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
        a = val_y[i]
        
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
ticks = np.linspace(-10,10,20)
plt.hist(rel_error, bins=ticks)
# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/potentials_rel_en.png',dpi=1200)
plt.show()

#%%

test = np.histogram(bar_error)
print(test[0])
ticks = np.linspace(-10,10,20)
plt.hist(bar_error, bins=ticks)
# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/potentials_rel_en.png',dpi=1200)
plt.show()

#%%

test = np.histogram(pos_error)
print(test[0])
ticks = np.linspace(-1,1,20)
plt.hist(pos_error, bins=ticks)
# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/potentials_rel_en.png',dpi=1200)
plt.show()

