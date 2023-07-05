#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 08:54:04 2021

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

np.set_printoptions(precision=2,suppress=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

#%%

directory_loc = '/Users/maximsecor/Desktop/'

print("\n----------------Welcome to QMML----------------")
print("\nLoading Data")
    
file_learning = directory_loc + 'ANN_1DTISE/Learning_curves.csv'
data_learning = pd.read_csv(file_learning,header=0)
features_learning = data_learning.values

print("Loading Data Complete")

#%%

learning_curves = (features_learning[:,:2])

#%%

print(learning_curves[79])

#%%

print(np.linspace(0,8000,80))

#%%

errors = np.array([0,0,0,-1,-3,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4])
magnitudes = np.array([2,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,3,3])

magnitudes_str = np.array([r'$10^{2}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10^{5}$',r'$10$',r'$10^{3}$',r'$10^{3}$'])
error_labels = np.array(["(kcal/mol)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(a.u.)","(kcal/mol)","(a.u.)","(a.u.)"])

print(magnitudes_str.shape)

#%%

# for i in range(18):
for i in range(18):

    data = learning_curves[(i*80+errors[i]):((i+1)*80+errors[i+1])]*(10**magnitudes[i])
    plt.plot((np.linspace(0,8000,len(data))),data)
    
    plt.rc('font', family='Helvetica')
    plt.rcParams["figure.figsize"] = (6,4)
    
    start = learning_curves[i*80+errors[i],1]*(10**magnitudes[i])
    end = learning_curves[(i+1)*80+errors[i+1]-1,0]*(10**magnitudes[i])
    step_size = (start-end)/5
    
    # print(start,end,step_size)
    
    temp_start = start
    temp_idx = 0
    temp_test = -1
    
    while temp_test < 0:
        
        temp_test = temp_start*(10**temp_idx) - 1
        
        # print(temp_start,temp_idx,temp_test)
        
        temp_idx = temp_idx + 1
        
    
    plt.xticks(np.arange(0, 9000, step=2000))
    plt.ylim(end-step_size/2,start)
    
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
    
    plt.xlabel('Traning Examples')
    plt.ylabel('MAE x'+magnitudes_str[i]+' '+error_labels[i])
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.show()

#%%

# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi);