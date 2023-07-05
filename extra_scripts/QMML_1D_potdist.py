#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:57:32 2020

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

from scipy.signal import argrelextrema

#%%

directory_loc = '/Users/maximsecor/Desktop/'

print("\n----------------Welcome to QMML----------------")
print("\nLoading Data")
    
file_potential = directory_loc + 'ANN_1DTISE/TRAINING_DATA/1d_potential.csv'
data_pot = pd.read_csv(file_potential)

potentials_referenced = data_pot.values

print("Loading Data Complete")

#%%

n_minima = []
rel_en = []
rel_pos = []
bar1 = []
bar2 = []
freq = []

for i in range(len(potentials_referenced)):
# for i in range(100):

    "Potential"
    a = potentials_referenced[i]
    
    "Minima"
    maxInd = argrelextrema(a, np.less)
    r = a[maxInd]
    if len(r)>1:
        n_minima.append(2)
        rel_en.append(r[0]-r[1])
        pos = np.array(maxInd)*(2/1024)-1
        rel_pos.append(pos[0,1]-pos[0,0])
    else:
        n_minima.append(1)
        
    double = np.gradient(np.gradient(a,2/1024),2/1024)
    for j in range(len(maxInd[0])):
        freq.append((np.sqrt((double[maxInd[0][j]]/(1.8897**2))/1836)*627*350))
        
    "Barriers"
    maxIndbar = argrelextrema(a, np.greater)
    rbar = a[maxIndbar]
    if len(rbar)==1:
        if r[0] > r[1]:
            bar1.append(rbar-r[1])
            bar2.append(rbar-r[0])
        else:
            bar1.append(rbar-r[0])
            bar2.append(rbar-r[1])
            
#%%
            
n_minima = np.array(n_minima)
test = np.histogram(n_minima, bins=[-0.5,0.5,1.5,2.5,3.5])
print(test[0])
plt.hist(n_minima, bins=2)


# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/potentials_singledoubleratio.png',dpi=1200)
plt.show()

#%%

plt.figure(figsize=(6,4))

rel_en = np.array(rel_en)
test = np.histogram(rel_en)
print(test[0])
plt.hist(-1*rel_en*627,bins=100)

plt.xlabel('Energy Difference Between Minima (kcal/mol)')
plt.ylabel('Counts')
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

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Dist1.tif',dpi=1200)

#%%

plt.figure(figsize=(6,4))

bar1 = np.array(bar1)
test = np.histogram(bar1)
print(test[0])
plt.hist(bar1*627,bins=100)

plt.xlabel('Energy of Barrier (kcal/mol)')
plt.ylabel('Counts')
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

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Dist2.tif',dpi=1200)

#%%

plt.figure(figsize=(6,4))

bar2 = np.array(bar2)
test = np.histogram(bar2)
print(test[0])
plt.hist(bar2*627,bins=100)

plt.xlabel('Energy Barrier (kcal/mol)')
plt.ylabel('Frequency')
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

# plt.savefig('/Users/maximsecor/Desktop/Models_ANN/potentials_barrier2.png',dpi=1200)
plt.show()

#%%

plt.figure(figsize=(6,4))

rel_pos = np.array(rel_pos)
test = np.histogram(rel_pos)
print(test[0])
plt.hist(rel_pos,bins=100)

plt.xlabel('Distance Between Minima (Å)')
plt.ylabel('Counts')
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

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Dist3.tif',dpi=1200)

#%%

plt.figure(figsize=(6,4))


freq = np.array(freq)
test = np.histogram(freq)
print(test[0])
plt.hist(freq,bins=100)

plt.xlabel('Frequencies at Minima (cm$^{-1}$)')
plt.ylabel('Counts')
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

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Dist4.tif',dpi=1200)

#%%

train_X, val_X, train_y, val_y = train_test_split(potentials_referenced, potentials_referenced, test_size = 0.1, random_state = 1103)

xlist = np.linspace(-0.5,1.5,1024)

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

plt.ylim(-1,20)

plt.plot(xlist,val_X[0]*627,'k')
plt.plot(xlist,val_X[65]*627,'b')
plt.plot(xlist,val_X[66]*627,'r')
plt.plot(xlist,val_X[50]*627,'m')

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/SamplePots.tif',dpi=1200)

