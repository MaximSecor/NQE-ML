#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:22:46 2021

@author: maximsecor
"""


#%%

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
# from xgboost import XGBRegressor

np.set_printoptions(precision=4,suppress=True)

import tensorflow as tf

from sklearn import preprocessing

import time
import os
import pandas as pd


# import seaborn as sns
import seaborn as sns; sns.set()

from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema

import matplotlib.animation as animation
from celluloid import Camera

directory_loc = '/Users/maximsecor/Desktop/'

#%%

@jit(nopython=True)
def fgh_2d(domain,potential,mass):
    
    nx = len(domain)
    dx = domain[1]-domain[0]
    k = np.pi/dx
    
    hmat = np.zeros((nx**2,nx**2))
    
    for xi in range(nx):
        for xj in range(nx):
            for yi in range(nx):
                for yj in range(nx):
                    if (xi == xj) & (yi == yj):
                        vmat = potential[xj,yj]
                        tmat = (k**2)*(2/3)
                    if (xi == xj) & (yi != yj):
                        dji = yj - yi
                        vmat = 0
                        tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                    if (xi != xj) & (yi == yj):
                        dji = xj - xi
                        vmat = 0
                        tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                    if (xi != xj) & (yi != yj):
                        vmat = 0
                        tmat = 0
                    hmat[xi+yi*nx,xj+yj*nx] = (1/(2*mass))*tmat+vmat
                        
    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln

#%%
    
potentials_2D = []

grid = 32

for i in range(2000):
    
    print(i)
    
    X2 = -2*np.random.random()-0.5
    X3 = -2*np.random.random()-0.5
    Y3 = -2*np.random.random()-0.5
    
    KX_1 = (3000*np.random.random()+1500)/(350*627)
    KX_2 = (3000*np.random.random()+1500)/(350*627)
    KX_3 = (3000*np.random.random()+1500)/(350*627)
    
    KY_1 = (3000*np.random.random()+1500)/(350*627)
    KY_2 = (3000*np.random.random()+1500)/(350*627)
    KY_3 = (3000*np.random.random()+1500)/(350*627)
    
    dE_2 = 2*np.random.random()/627
    dE_3 = 2*np.random.random()/627+dE_2
    
    xlist = np.linspace(-0.5,1.5,grid)*1.8897
    ylist = np.linspace(-0.5,1.5,grid)*1.8897
    
    X, Y = np.meshgrid(xlist,ylist)
    
    potential_1 = 0.5*KX_1*(X)**2 + 0.5*KY_1*Y**2 
    potential_2 = 0.5*KX_2*(X+X2)**2 + 0.5*KY_2*Y**2 + dE_2
    potential_3 = 0.5*KX_3*(X+X3)**2 + 0.5*KY_3*(Y+Y3)**2 + dE_3
    
    couplings = np.full((grid,grid),1)*(10/627)
    
    two_state = np.array([[potential_1,couplings,couplings],[couplings,potential_2,couplings],[couplings,couplings,potential_3]])
    
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES = ground_state_PES - np.min(ground_state_PES)
    
    potentials_2D.append(ground_state_PES)
    
    # plt.contourf(xlist,ylist,ground_state_PES*627,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
    # plt.show()

potentials_2D = np.array(potentials_2D)

#%%

start = time.time()

target_potentials =[]
target_wavefunctions = []
target_energies = []

for i in range(2000):

    target_potentials.append(potentials_2D[i].reshape(grid**2))
    
    temp = fgh_2d(xlist,potentials_2D[i],1836)
    target_wavefunctions.append(temp[1].T[0:5].T)
    target_energies.append(temp[0][0:50])
    
    print(i)
    
target_potentials = np.array(target_potentials)
target_wavefunctions = np.array(target_wavefunctions)
target_energies = np.array(target_energies)

end = time.time()
print(end-start)

#%%

print(target_wavefunctions.reshape(target_wavefunctions.shape[0],target_wavefunctions.shape[1]*target_wavefunctions.shape[2]).shape)

#%%

file_potentials = '/Users/maximsecor/Desktop/ANN_2D/potentials.csv'
file_wavefunctions = '/Users/maximsecor/Desktop/ANN_2D/wavefunctions.csv'
file_energies = '/Users/maximsecor/Desktop/ANN_2D/energies.csv'

os.system('touch ' + file_potentials)
os.system('touch ' + file_wavefunctions)
os.system('touch ' + file_energies)

df_features_potentials = pd.DataFrame(target_potentials)
df_target_wavefunctions = pd.DataFrame(target_wavefunctions.reshape(target_wavefunctions.shape[0],target_wavefunctions.shape[1]*target_wavefunctions.shape[2]))
df_target_energies = pd.DataFrame(target_energies)

df_features_potentials.to_csv(file_potentials, index = False, header=True)
df_target_wavefunctions.to_csv(file_wavefunctions, index = False, header=True)
df_target_energies.to_csv(file_energies, index = False, header=True)

#%%

file_potentials = '/Users/maximsecor/Desktop/ANN_2D/potentials.csv'
file_wavefunctions = '/Users/maximsecor/Desktop/ANN_2D/wavefunctions.csv'
file_energies = '/Users/maximsecor/Desktop/ANN_2D/energies.csv'

data_potentials = pd.read_csv(file_potentials)
data_wavefunctions = pd.read_csv(file_wavefunctions)
data_energies = pd.read_csv(file_energies)

potentials = data_potentials.values
wavefunctions = data_wavefunctions.values
energies = data_energies.values

grid =32
wavefunctions = (wavefunctions.reshape(2000,1024,5))
xlist = np.linspace(-0.5,1.5,grid)*1.8897
ylist = np.linspace(-0.5,1.5,grid)*1.8897
X, Y = np.meshgrid(xlist,ylist)

#%%

""" POTENTIAL --> ENERGIES """

#%%

train = potentials
target = energies[:,:5]*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

NN_model = Sequential()
NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.0001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

start = time.time()

for i in range(6):
    NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)

end = time.time()
print(end-start)

#%%

NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_pot_5nrg')
    
#%%
   
NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_pot_5nrg')

train = potentials
target = energies[:,:5]*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
 
#%%

print(np.min(val_y[:,0]),np.max(val_y[:,0]))
print(np.min(val_y[:,4]),np.max(val_y[:,4]))
print(np.min(val_y[:,:5]),np.max(val_y[:,:5]))

#%%

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

"Error in energies and excitation energies"

print("Energy Errors")
energy_error = predictions_val - val_y

print(np.mean(abs(energy_error)))

for i in range(5):
    print("Energy Error of State "+str(int(i))+": ",np.mean(np.abs(energy_error[:,i])))

#%%

"Error Histogram in wavenumbers"

energy_error_flat = energy_error.reshape(energy_error.shape[0]*energy_error.shape[1])
# print(energy_error_flat)

ticks = np.linspace(-0.25,0.25,100)
plt.xticks(np.arange(-0.2, 0.3, step=0.1))
plt.hist(energy_error_flat, bins=ticks) 
plt.rcParams["figure.figsize"] = (4,4)

plt.ylabel('Counts')
plt.xlabel('ANN Error (kcal/mol)')

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_energies_hist_2D.tiff',dpi=1200)

#%%

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_val)))

for i in range(5):
    plot0[0] = val_y[:,i]
    plot0[1] = predictions_val[:,i]
    data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
    ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('ANN Energy (kcal/mol)')
ax.set_xlabel('FGH Energy (kcal/mol)')
# ax.set_title('Energy Comaprison')

plt.rc('font', family='Helvetica')
plt.axes().set_aspect('equal')
plt.rcParams["figure.figsize"] = (4,4)
plt.xticks(np.arange(0, 12, step=2))
plt.yticks(np.arange(0, 12, step=2))

x = np.linspace(0,np.max(val_y[:,i])+1,1000)
y = x
plt.plot(x, y, '--k', linewidth=1.5)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_energies_2D.tiff',dpi=1200)

#%%

train = potentials
target = energies*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

NN_model = Sequential()
NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.0001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

start = time.time()

for i in range(6):
    NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)

end = time.time()
print(end-start)

NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_pot_50nrg')
    
#%%
   
NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_pot_50nrg')
 
train = potentials
target = energies*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

#%%

print(np.min(val_y),np.max(val_y))
print(np.min(val_y[:,0]),np.max(val_y[:,0]))
print(np.min(val_y[:,49]),np.max(val_y[:,49]))
print(np.min(val_y[:,:5]),np.max(val_y[:,:5]))

#%%

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

"Error in energies and excitation energies"

print("Energy Errors")
energy_error = predictions_val - val_y

print(np.mean(abs(energy_error)))

test = 0
for i in range(5):
    print("Energy Error of State "+str(int(i))+": ",np.mean(np.abs(energy_error[:,i])))
    test = test + np.mean(np.abs(energy_error[:,i]))

print(test/5)

#%%

"Error Histogram in wavenumbers"

energy_error_flat = energy_error.reshape(energy_error.shape[0]*energy_error.shape[1])
print(energy_error_flat)

ticks = np.linspace(-0.25,0.25,100)
plt.xticks(np.arange(-0.2, 0.3, step=0.1))
plt.hist(energy_error_flat, bins=ticks) 
plt.rcParams["figure.figsize"] = (4,4)

plt.ylabel('Counts')
plt.xlabel('ANN Error (kcal/mol)')



plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_energies_hist_2D.tiff',dpi=1200)

#%%

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_val)))

for i in range(50):
    plot0[0] = val_y[:,i]
    plot0[1] = predictions_val[:,i]
    data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
    ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('ANN Energy (kcal/mol)')
ax.set_xlabel('FGH Energy (kcal/mol)')
# ax.set_title('Energy Comaprison')

plt.rc('font', family='Helvetica')
plt.axes().set_aspect('equal')
plt.rcParams["figure.figsize"] = (4,4)
plt.xticks(np.arange(0, 24, step=4))
plt.yticks(np.arange(0, 24, step=4))

x = np.linspace(0,np.max(val_y[:,i])+1,1000)
y = x
plt.plot(x, y, '--k', linewidth=1.5)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_energies_2D.tiff',dpi=1200)

#%%

""" POTENTIAL --> WAVEFUNCTION """

#%%

def custom_loss(y_true, y_pred):

    overlap = K.sum(y_true*y_pred,1)
    loss_1 = K.sum(K.square(1-K.square(overlap)))

    self_overlap = K.sum(y_pred*y_pred,1)
    loss_2 = K.sum(K.square(1-self_overlap))

    loss = loss_1 + loss_2

    return loss

#%%

for q in range(5):
    
    train = potentials
    target = wavefunctions[:,:,q]
    
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)
    
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    
    NN_model = Sequential()
    NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
    for layers in range(3):
        NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
    
    opt = Adam(learning_rate=(0.0001))
    NN_model.compile(loss=custom_loss, optimizer=opt, metrics=['mean_absolute_error'])
    
    for i in range(6):
        NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)
    
    NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_pot_'+str(q)+'wave')

#%%

MAE_cross = []
overlap_tracker = []

for q in range(5):

    print('\n')
    NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_pot_'+str(q)+'wave', custom_objects={'custom_loss':custom_loss})
    
    train = potentials
    target = wavefunctions[:,:,q]
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    
    predictions_val = NN_model.predict(val_X_tf)
    MAE = mean_absolute_error(val_y**2, predictions_val**2)
    print('Cross-Validation Set Error =', MAE)
    MAE_cross.append(MAE)
    
    predictions_train = NN_model.predict(train_X_tf)
    MAE = mean_absolute_error(train_y_tf**2, predictions_train**2)
    print('Training Set Error =', MAE)

    val_y_norm = (val_y.T*np.sqrt(1/np.sum(val_y**2,1)).T).T
    # print(np.sum(val_y_norm**2,1))

    predictions_val_smooth = (predictions_val.T*np.sqrt(1/np.sum(predictions_val**2,1)).T).T
    # print(np.sum(predictions_val_smooth**2,1))
    
    overlaps = np.abs(np.sum(val_y*predictions_val_smooth,1))
    overlap_tracker.append(np.mean(np.abs(overlaps)))
    print(np.mean(np.abs(overlaps)))
    print(np.min(overlaps),np.max(overlaps))

print('\n')
MAE_cross = np.array(MAE_cross)
average_MAE = np.mean(MAE_cross)
print('average_MAE: ', average_MAE)

overlap_tracker = np.array(overlap_tracker)
overlap_tracker = np.mean(overlap_tracker)
print('overlap_tracker: ', overlap_tracker)

#%%

# for i in range(10):
    
#     true = val_y[i].reshape(32,32)
#     plt.contourf(xlist,ylist,true,levels = [-0.20,-0.175,-0.15,-0.125,-0.10,-0.075,-0.05,-0.025,0.00,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20])
#     plt.show()

#     pred = predictions_val[i].reshape(32,32)
#     plt.contourf(xlist,ylist,pred,levels = [-0.20,-0.175,-0.15,-0.125,-0.10,-0.075,-0.05,-0.025,0.00,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20])
#     plt.show()
    
#     err = true-pred
#     plt.contourf(xlist,ylist,err,levels = [-0.20,-0.175,-0.15,-0.125,-0.10,-0.075,-0.05,-0.025,0.00,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20])
#     plt.show()
    
#     test = val_X[i].reshape(32,32).T
#     plt.contourf(xlist,ylist,test*627,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
#     plt.colorbar()
#     plt.show()
    
#%%
    
for k in range(5):

    print('\n')
    NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_pot_'+str(k)+'wave', custom_objects={'custom_loss':custom_loss})
    
    train = potentials
    target = wavefunctions[:,:,k]
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    
    predictions_val = NN_model.predict(val_X_tf)
    val_y_norm = (val_y.T*np.sqrt(1/np.sum(val_y**2,1)).T).T
    predictions_val_smooth = (predictions_val.T*np.sqrt(1/np.sum(predictions_val**2,1)).T).T
    predictions_val_smooth[predictions_val_smooth**2<0.00001]=0
    
    grid = 32
    xlist = np.linspace(-0.5,1.5,grid)*1.8897
    ylist = np.linspace(-0.5,1.5,grid)*1.8897
    X, Y = np.meshgrid(xlist,ylist)
    
    # test_error = 0.005
    # error = np.mean(np.abs(predictions_val_smooth-val_y),1)
    # best = np.where((error<test_error)==True)[0]

    # while (len(best))>10:
    #     test_error = test_error-0.00001
    #     error = np.mean(np.abs(predictions_val_smooth-val_y),1)
    #     best = np.where((error<test_error)==True)[0]
    #     # print(test_error,best.shape)
    #     # print(error)
        
    # print(best.shape)
    
    if k == 0:
        best = np.array([15,77,163,173])
    if k == 1:
        best = np.array([191,172,153,169])
    if k == 2:
        best = np.array([98,165,199,153,127])
    if k == 3:
        best = np.array([41,40,53,37,72,131])
    if k == 4:
        best = np.array([16,87,184,198,40,148])
    
    for q in range(len(best)):
        
        # print(len(best))
        # print(np.random.randint(len(best)))
        
        i = best[q]
        
        # true = val_y[i].reshape(32,32).T
        # plt.contourf(xlist,ylist,true,levels = [-0.20,-0.175,-0.15,-0.125,-0.10,-0.075,-0.05,-0.025,0.00,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20])
        # plt.show()
    
        pred = predictions_val_smooth[i].reshape(32,32).T
        # plt.contourf(xlist,ylist,pred,levels = [-0.20,-0.175,-0.15,-0.125,-0.10,-0.075,-0.05,-0.025,0.00,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20])
        plt.contourf(xlist/1.889,ylist/1.889,pred)
        # plt.show()
        
        
        # err = true-pred
        # plt.contourf(xlist,ylist,err,levels = [-0.20,-0.175,-0.15,-0.125,-0.10,-0.075,-0.05,-0.025,0.00,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20])
        # plt.show()
        
        CS = plt.contour(X/1.889, Y/1.889, val_X[i].reshape(32,32)*627,levels = [-1,0,1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
        plt.clabel(CS, inline=1, fontsize=18, fmt = '%1.0f')
        
        plt.rc('font', family='Helvetica')
        plt.axes().set_aspect('equal')
        plt.rcParams["figure.figsize"] = (4,4)
        plt.xticks(np.arange(-0.25, 1.75, step=0.5))
        plt.yticks(np.arange(-0.25, 1.75, step=0.5))
        
        plt.ylabel('Position (Å)')
        plt.xlabel('Position (Å)')
        
        SMALL_SIZE = 18
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 20
        
        plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        # plt.title('State: '+str(k)+' & Example: '+str(i))
        
        plt.tick_params(bottom=True, top=False, left=True, right=False)
        
        plt.show()
    
#%%

""" POTENTIAL --> DENSITIES """

#%%
        
for q in range(5):
    
    train = potentials
    target = wavefunctions[:,:,q]**2
    
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)
    
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    
    NN_model = Sequential()
    NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
    for layers in range(3):
        NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
    
    opt = Adam(learning_rate=(0.0001))
    NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
    
    for i in range(6):
        NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)
    
    NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_pot_'+str(q)+'den')
   
#%%

test = 0 

for q in range(5):

    print('\n')
    NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_pot_'+str(q)+'den')
    
    train = potentials
    target = wavefunctions[:,:,q]**2
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    
    predictions_val = NN_model.predict(val_X_tf)
    MAE = mean_absolute_error(val_y, predictions_val)
    print('Cross-Validation Set Error =', MAE)
    test = test + MAE
    
    predictions_train = NN_model.predict(train_X_tf)
    MAE = mean_absolute_error(train_y_tf, predictions_train)
    print('Training Set Error =', MAE)

    val_y_norm = (val_y.T*(1/np.sum(val_y,1)).T).T
    # print(np.sum(val_y_norm,1))

    predictions_val_smooth = (predictions_val.T*(1/np.sum(predictions_val,1)).T).T
    # print(np.sum(predictions_val_smooth,1))
   
print(test/5)
    
#%%
    
for k in range(5):

    print('\n')
    NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_pot_'+str(k)+'den')
    
    train = potentials
    target = wavefunctions[:,:,k]**2
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    
    predictions_val = NN_model.predict(val_X_tf)
    predictions_val_smooth = (predictions_val.T*(1/np.sum(predictions_val,1)).T).T
    predictions_val_smooth[predictions_val_smooth<0.00000001]=0
    
    grid = 32
    xlist = np.linspace(-0.5,1.5,grid)
    ylist = np.linspace(-0.5,1.5,grid)
    X, Y = np.meshgrid(xlist,ylist)
    
    # test_error = 0.0004
    # error = np.mean(np.abs(predictions_val_smooth-val_y),1)
    # best = np.where((error<test_error)==True)[0]

    # while (len(best))>25:
    #     test_error = test_error-0.000001
    #     error = np.mean(np.abs(predictions_val_smooth-val_y),1)
    #     best = np.where((error<test_error)==True)[0]
    #     # print(test_error,best.shape)
    #     # print(error)
        
    # print(test_error,best.shape)
    
    # error = np.mean(np.abs(predictions_val_smooth-val_y),1)
    # best = np.where((error<0.0001)==True)[0]
    # print(best.shape)
    # # print(error)
    
    
    if k == 0:
        best = np.array([43])
    if k == 1:
        best = np.array([100,190])
    if k == 2:
        best = np.array([])
    if k == 3:
        best = np.array([])
    if k == 4:
        best = np.array([169])
    
    
    for q in range(len(best)):
        
        i = best[q]
        pred = predictions_val_smooth[i].reshape(32,32).T
        plt.contourf(xlist,ylist,pred)
        
        CS = plt.contour(X, Y, val_X[i].reshape(32,32)*627,levels = [-1,0,1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
        plt.clabel(CS, inline=1, fontsize=18, fmt = '%1.0f')
        
        plt.rc('font', family='Helvetica')
        plt.axes().set_aspect('equal')
        6
        plt.xticks(np.arange(-0.25, 1.75, step=0.5))
        plt.yticks(np.arange(-0.25, 1.75, step=0.5))
        
        plt.ylabel('Position (Å)')
        plt.xlabel('Position (Å)')
        
        SMALL_SIZE = 18
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 20
        
        plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        # plt.title('State: '+str(k)+' & Example: '+str(i))
        
        plt.tick_params(bottom=True, top=False, left=True, right=False)
        
        plt.show()
        
#%%
    
""" GS DENSITY --> DENSITIES """

#%%
        
for q in range(4):
    
    q = q + 1
    
    train = wavefunctions[:,:,0]**2
    target = wavefunctions[:,:,q]**2
    
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)
    
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    
    NN_model = Sequential()
    NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
    for layers in range(3):
        NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
    
    opt = Adam(learning_rate=(0.0001))
    NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
    
    for i in range(6):
        NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)
    
    NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_den_'+str(q)+'den')
   
#%%

test = 0

for q in range(4):

    q = q + 1
    
    print('\n')
    NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_den_'+str(q)+'den')
    
    train = wavefunctions[:,:,0]**2
    target = wavefunctions[:,:,q]**2
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    
    predictions_val = NN_model.predict(val_X_tf)
    MAE = mean_absolute_error(val_y, predictions_val)
    print('Cross-Validation Set Error =', MAE)
    test = test + MAE
    
    predictions_train = NN_model.predict(train_X_tf)
    MAE = mean_absolute_error(train_y_tf, predictions_train)
    print('Training Set Error =', MAE)

    val_y_norm = (val_y.T*(1/np.sum(val_y,1)).T).T
    # print(np.sum(val_y_norm**2,1))

    predictions_val_smooth = (predictions_val.T*(1/np.sum(predictions_val,1)).T).T
    # print(np.sum(predictions_val_smooth**2,1))
    
print(test/4)
    
#%%
    
for k in range(4):
    
    k = k + 1

    print('\n')
    NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_den_'+str(k)+'den')
    
    train = wavefunctions[:,:,0]**2
    target = wavefunctions[:,:,k]**2
    
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    
    predictions_val = NN_model.predict(val_X_tf)
    predictions_val_smooth = (predictions_val.T*(1/np.sum(predictions_val,1)).T).T
    predictions_val_smooth[predictions_val_smooth**2<0.000001]=0
    
    grid = 32
    xlist = np.linspace(-0.5,1.5,grid)*1.8897
    ylist = np.linspace(-0.5,1.5,grid)*1.8897
    X, Y = np.meshgrid(xlist,ylist)
    
    # test_error = 0.001
    # error = np.mean(np.abs(predictions_val_smooth-val_y),1)
    # best = np.where((error<test_error)==True)[0]

    # while (len(best))>5:
    #     test_error = test_error-0.00001
    #     error = np.mean(np.abs(predictions_val_smooth-val_y),1)
    #     best = np.where((error<test_error)==True)[0]
    #     # print(test_error,best.shape)
    #     # print(error)
        
    # print(test_error,best.shape)
    

    if k == 1:
        best = np.array([10,70,76])
    if k == 2:
        best = np.array([])
    if k == 3:
        best = np.array([147,163])
    if k == 4:
        best = np.array([169])
    
    # for q in range(len(best)):
    for q in range(len(best)):
        
        i = best[q]
    
        pred = predictions_val_smooth[i].reshape(32,32).T
        plt.contourf(xlist/1.889,ylist/1.889,pred)
        
        CS = plt.contour(X/1.889, Y/1.889, val_X[i].reshape(32,32).T*627,levels = [-1,0,1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
        plt.clabel(CS, inline=1, fontsize=0)
        
        plt.rc('font', family='Helvetica')
        plt.axes().set_aspect('equal')
        plt.rcParams["figure.figsize"] = (4,4)
        plt.xticks(np.arange(-0.25, 1.75, step=0.5))
        plt.yticks(np.arange(-0.25, 1.75, step=0.5))
        
        plt.ylabel('Position (Å)')
        plt.xlabel('Position (Å)')
        
        SMALL_SIZE = 18
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 20
        
        plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        # plt.title('State: '+str(k)+' & Example: '+str(i))
        
        plt.show()
        
#%%
    
""" GS DENSITY --> SPECTRA """

#%%

train = wavefunctions[:,:,0]**2
target = energies[:,:5]*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

NN_model = Sequential()
NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.0001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

for i in range(6):
    NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)

NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_den_5nrg')
    
#%%
   
NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_den_5nrg')

train = wavefunctions[:,:,0]**2
target = energies[:,:5]*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
 
#%%

print(np.min(val_y[:,:5]),np.max(val_y[:,:5]))

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

"Error in energies and excitation energies"

print("Energy Errors")
energy_error = predictions_val - val_y

print(np.mean(abs(energy_error)))

for i in range(5):
    print("Energy Error of State "+str(int(i))+": ",np.mean(np.abs(energy_error[:,i])))

#%%

"Error Histogram in wavenumbers"

energy_error_flat = energy_error.reshape(energy_error.shape[0]*energy_error.shape[1])
# print(energy_error_flat)

ticks = np.linspace(-0.25,0.25,100)
plt.xticks(np.arange(-0.2, 0.3, step=0.1))
plt.hist(energy_error_flat, bins=ticks) 
plt.rcParams["figure.figsize"] = (4,4)

plt.ylabel('Counts')
plt.xlabel('ANN Error (kcal/mol)')

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_energies_hist_2D.tiff',dpi=1200)

#%%

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_val)))

for i in range(5):
    plot0[0] = val_y[:,i]
    plot0[1] = predictions_val[:,i]
    data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
    ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('ANN Energy (kcal/mol)')
ax.set_xlabel('FGH Energy (kcal/mol)')
# ax.set_title('Energy Comaprison')

plt.rc('font', family='Helvetica')
plt.axes().set_aspect('equal')
plt.rcParams["figure.figsize"] = (4,4)
plt.xticks(np.arange(0, 12, step=2))
plt.yticks(np.arange(0, 12, step=2))

x = np.linspace(0,np.max(val_y[:,i])+1,1000)
y = x
plt.plot(x, y, '--k', linewidth=1.5)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_energies_2D.tiff',dpi=1200)

#%%

train = wavefunctions[:,:,0]**2
target = energies*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

NN_model = Sequential()
NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.0001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

for i in range(6):
    NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)

NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_den_50nrg')
    
#%%
   
NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_den_50nrg')

train = wavefunctions[:,:,0]**2
target = energies*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
 
#%%

print(np.min(val_y),np.max(val_y))

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

"Error in energies and excitation energies"

print("Energy Errors")
energy_error = predictions_val - val_y

print(np.mean(abs(energy_error)))

for i in range(50):
    print("Energy Error of State "+str(int(i))+": ",np.mean(np.abs(energy_error[:,i])))

#%%

"Error Histogram in wavenumbers"

energy_error_flat = energy_error.reshape(energy_error.shape[0]*energy_error.shape[1])
# print(energy_error_flat)

ticks = np.linspace(-1,1,100)
plt.xticks(np.arange(-1, 1.4, step=0.4))

plt.hist(energy_error_flat, bins=ticks) 
plt.rcParams["figure.figsize"] = (4,4)

plt.ylabel('Counts')
plt.xlabel('ANN Error (kcal/mol)')

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_energies_hist_2D.tiff',dpi=1200)

#%%

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_val)))

for i in range(50):
    plot0[0] = val_y[:,i]
    plot0[1] = predictions_val[:,i]
    data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
    ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('ANN Energy (kcal/mol)')
ax.set_xlabel('FGH Energy (kcal/mol)')
# ax.set_title('Energy Comaprison')

plt.rc('font', family='Helvetica')
plt.axes().set_aspect('equal')
plt.rcParams["figure.figsize"] = (4,4)
plt.xticks(np.arange(0, 20, step=4))
plt.yticks(np.arange(0, 20, step=4))

x = np.linspace(0,np.max(val_y[:,i])+1,1000)
y = x
plt.plot(x, y, '--k', linewidth=1.5)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_energies_2D.tiff',dpi=1200)

#%%
    
""" SPECTRA --> POTENTIAL """

#%%

train = energies*627
target = potentials*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

NN_model = Sequential()
NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.0001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

for i in range(6):
    NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)

NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_50nrg_pot')
    
#%%
   
NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_50nrg_pot')

train = energies*627
target = potentials*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

#%%

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

error = np.mean(np.abs(predictions_val-val_y),1)
best = np.where((error<0.5)==True)[0]
print(best)

#%%

spectra = np.array([115,103,125,98])

for i in range(len(spectra)):
    predictedY = val_X[spectra[i],:10]
    s = np.full(len(predictedY),200)
    
    plt.scatter(np.zeros_like(predictedY),predictedY,s,marker="_")
    plt.rcParams["figure.figsize"] = (0.25,4)
    
    plt.xticks([])
    
    plt.yticks(np.arange(0.0, 10, step=2))
    
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
    
    # plt.rcParams['axes.facecolor'] = 'w'
    # plt.set_marker(self, |)
    
    plt.show()
    
#%%
    
# test_error = 0.001
# error = np.mean(np.abs(predictions_val_smooth-val_y),1)
# best = np.where((error<test_error)==True)[0]

# while (len(best))>5:
#     test_error = test_error-0.00001
#     error = np.mean(np.abs(predictions_val_smooth-val_y),1)
#     best = np.where((error<test_error)==True)[0]
#     # print(test_error,best.shape)
#     # print(error)
    
# print(test_error,best.shape)
    

test_error = 1
error = np.mean(np.abs(predictions_val-val_y),1)
best = np.where((error<test_error)==True)[0]


while (len(best))>15:
    test_error = test_error-0.01
    error = np.mean(np.abs(predictions_val-val_y),1)
    best = np.where((error<test_error)==True)[0]
    # print(test_error,best.shape)
    # print(error)
    
print(test_error,best.shape)

best = np.array([115,103,125,98])

for i in range(len(best)):
# for i in range(5):
    
    print(best[i])
    
    # test = val_X[best[i]].reshape(32,32).T
    # plt.contourf(xlist,ylist,test)
    
    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])
    
    # plt.show()
    
    CS = plt.contour(X/1.889, Y/1.889, val_y[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='w', linestyles="dotted")
    # plt.clabel(CS1, inline=1, fontsize=18, fmt = '%1.0f')
    
    CS2 = plt.contour(X/1.889, Y/1.889, predictions_val[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='cyan', linestyles="solid")
    plt.clabel(CS2, inline=1, fontsize=18, fmt = '%1.0f')
    
    # plt.title('Example: ' + str(best[i]))
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', family='Helvetica')
    plt.axes().set_aspect('equal')
    plt.rcParams["figure.figsize"] = (4,4)
    plt.xticks(np.arange(-0.25, 1.75, step=0.5))
    plt.yticks(np.arange(-0.25, 1.75, step=0.5))
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.rcParams['axes.facecolor'] = 'black'
    
    plt.grid(False)
    # plt.right_ax.grid(False)
    plt.tick_params(bottom=True, top=False, left=True, right=False)
    
    plt.ylabel('Position (Å)')
    plt.xlabel('Position (Å)')  
    
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
    
    plt.show()
    
    # predictedY = val_X[i,:10]*627
    # plt.scatter(predictedY, np.zeros_like(predictedY))
    # plt.rcParams["figure.figsize"] = (4,0.25)
    # plt.xticks(np.arange(0.0, 10, step=2))
    # plt.yticks([])
    # plt.show()
    
#%%
    
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

#%%

for i in range(len(best)):
# # for i in range(5):
        
    q = best[i]
    predictedY = val_X[q,:10]
    s = np.full(len(predictedY),200)
    
    plt.scatter(np.zeros_like(predictedY),predictedY,s,marker="_")
    plt.rcParams["figure.figsize"] = (0.25,4)
    plt.xticks([])
    plt.yticks(np.arange(0.0, 8, step=2))
    # plt.rcParams['axes.facecolor'] = 'w'
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', family='Helvetica')
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
    
    # plt.set_marker(self, |)
    plt.show()

#%%
    
""" GS DENSITY --> POTENTIAL """

#%%

train = wavefunctions[:,:,0]**2
target = potentials*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

NN_model = Sequential()
NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.0001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

for i in range(6):
    NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)

NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_gsden_pot')
    
#%%
   
NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_gsden_pot')

train = wavefunctions[:,:,0]**2
target = potentials*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

#%%

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

error = np.mean(np.abs(predictions_val-val_y),1)
best = np.where((error<0.5)==True)[0]
print(best)

#%%

test_error = 1
error = np.mean(np.abs(predictions_val-val_y),1)
best = np.where((error<test_error)==True)[0]


while (len(best))>5:
    test_error = test_error-0.01
    error = np.mean(np.abs(predictions_val-val_y),1)
    best = np.where((error<test_error)==True)[0]
    # print(test_error,best.shape)
    # print(error)
    
print(test_error,best.shape)

best = np.array([152,120,119,96,86,73,49,28])

for i in range(len(best)):
    
    print(best[i])
    
    test = val_X[best[i]].reshape(32,32).T
    plt.contourf(xlist/1.889,ylist/1.889,test)
    
    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])
    
    # plt.show()
    
    CS = plt.contour(X/1.889, Y/1.889, val_y[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='w', linestyles="solid",linewidths=1.5)
    # plt.clabel(CS, inline=1, fontsize=14, fmt = '%1.0f')
    
    CS2 = plt.contour(X/1.889, Y/1.889, predictions_val[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='cyan', linestyles="solid")
    plt.clabel(CS2, inline=1, fontsize=18, fmt = '%1.0f')

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', family='Helvetica')
    plt.axes().set_aspect('equal')
    plt.rcParams["figure.figsize"] = (4,4)
    plt.xticks(np.arange(-0.25, 1.75, step=0.5))
    plt.yticks(np.arange(-0.25, 1.75, step=0.5))
    
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
    
    plt.tick_params(bottom=True, top=False, left=True, right=False)
    
    plt.ylabel('Position (Å)')
    plt.xlabel('Position (Å)')  
    # plt.title('Example: ' + str(best[i]))
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.rcParams['figure.dpi'] = 300

    plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/2d_den_pot_'+str(i)+'t.png',dpi=600,bbox_inches='tight')
    plt.show()
    
#%%

# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300

i = 2
print(best[i])
    
test = val_X[best[i]].reshape(32,32).T
plt.contourf(xlist/1.889,ylist/1.889,test)

# pred = predictions_val[i].reshape(32,32)
# plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])

# plt.show()

CS = plt.contour(X/1.889, Y/1.889, val_y[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='w', linestyles="dotted",linewidths=1.5)
# plt.clabel(CS, inline=1, fontsize=14, fmt = '%1.0f')

CS2 = plt.contour(X/1.889, Y/1.889, predictions_val[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='cyan', linestyles="solid")
plt.clabel(CS2, inline=1, fontsize=18, fmt = '%1.0f')



SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', family='Helvetica')
plt.axes().set_aspect('equal')
plt.rcParams["figure.figsize"] = (4,4)
plt.xticks(np.arange(-0.25, 1.75, step=0.5))
plt.yticks(np.arange(-0.25, 1.75, step=0.5))

plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True

plt.tick_params(bottom=True, top=False, left=True, right=False)

plt.ylabel('Position (Å)')
plt.xlabel('Position (Å)')  
# plt.title('Example: ' + str(best[i]))

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()

# plt.savefig(directory_loc + 'ANN_1DTISE/PES_spectra/2d_den_pot.png',dpi=1200)

#%%
    
best = np.array([152,120,119,96,86,73,49,28])

for i in range(len(best)):
    
    print(best[i])
    
    test = val_X[best[i]].reshape(32,32).T
    plt.contourf(xlist/1.889,ylist/1.889, predictions_val[best[i]].reshape(32,32),levels = [-1.0,1,2,4,6,10,15,20,30,40])
    
    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])
    
    # plt.show()
    
    CS = plt.contour(X/1.889, Y/1.889, test, levels=4, colors='w', linestyles="dotted")
    # plt.clabel(CS, inline=1, fontsize=0, fmt = '%1.0f')
    
    CS2 = plt.contour(X/1.889, Y/1.889, val_y[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='cyan', linestyles="solid")
    plt.clabel(CS2, inline=1, fontsize=14, fmt = '%1.0f')

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', family='Helvetica')
    plt.axes().set_aspect('equal')
    plt.rcParams["figure.figsize"] = (4,4)
    plt.xticks(np.arange(-0.25, 1.75, step=0.5))
    plt.yticks(np.arange(-0.25, 1.75, step=0.5))
    
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
    
    plt.tick_params(bottom=True, top=False, left=True, right=False)
    
    plt.ylabel('Position (Å)')
    plt.xlabel('Position (Å)')  
    # plt.title('Example: ' + str(best[i]))
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.show()

#%%
    
i =0
print(best[i])

# pred = predictions_val[i].reshape(32,32)
# plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])

# plt.show()

CS = plt.contour(X/1.889, Y/1.889, test, levels=4, colors='red', linestyles="dotted")
# plt.clabel(CS, inline=1, fontsize=0, fmt = '%1.0f')

CS2 = plt.contour(X/1.889, Y/1.889, val_y[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='cyan', linestyles="solid")
plt.clabel(CS2, inline=1, fontsize=14, fmt = '%1.0f')

test = val_X[best[i]].reshape(32,32).T
plt.contourf(xlist/1.889,ylist/1.889, predictions_val[best[i]].reshape(32,32),levels = [-1.0,1,2,4,6,10,15,20,30,40],cmaps='plasma_r')


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', family='Helvetica')
plt.axes().set_aspect('equal')
plt.rcParams["figure.figsize"] = (4,4)
plt.xticks(np.arange(-0.25, 1.75, step=0.5))
plt.yticks(np.arange(-0.25, 1.75, step=0.5))

plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True

plt.tick_params(bottom=True, top=False, left=True, right=False)

plt.ylabel('Position (Å)')
plt.xlabel('Position (Å)')  
# plt.title('Example: ' + str(best[i]))

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()

#%%


for i in range(10):
    
    # true = val_y[i].reshape(32,32)
    # plt.contourf(xlist,ylist,true,levels = [-1,0,2,4,6,10,15,20,30])
    # plt.show()

    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,2,4,6,10,15,20,30])
    # plt.show()
    
    # err = true-pred
    # plt.contourf(xlist,ylist,err,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
    # plt.show()

    test = val_X[i].reshape(32,32).T
    plt.contourf(xlist,ylist,test)
    
    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])
    
    CS2 = plt.contour(X, Y, predictions_val[i].reshape(32,32),levels = [-1,0,1,2,4,6,10,15,20,30], colors='r', linestyles="solid")
    plt.clabel(CS2, inline=1, fontsize=10)
    # plt.show()
    
    CS = plt.contour(X, Y, val_y[i].reshape(32,32),levels = [-1,0,1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()
    
    # test = val_X[i].reshape(32,32).T
    # plt.contourf(xlist,ylist,test*627,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
    # plt.colorbar()
    # plt.show()

#%%

train = wavefunctions[:,:,4]**2
target = potentials*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

NN_model = Sequential()
NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.0001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

for i in range(6):
    NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)

NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_4esden_pot')
    
#%%
   
NN_model = load_model('/Users/maximsecor/Desktop/ANN_2D/model_4esden_pot')

train = wavefunctions[:,:,4]**2
target = potentials*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

#%%

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

error = np.mean(np.abs(predictions_val-val_y),1)
best = np.where((error<0.5)==True)[0]
print(best)

#%%
test_error = 1
error = np.mean(np.abs(predictions_val-val_y),1)
best = np.where((error<test_error)==True)[0]


while (len(best))>10:
    test_error = test_error-0.01
    error = np.mean(np.abs(predictions_val-val_y),1)
    best = np.where((error<test_error)==True)[0]
    # print(test_error,best.shape)
    # print(error)
    
print(test_error,best.shape)

# best = np.array([152,120,119,96,86,73,49,28])

for i in range(len(best)):
    
    print(best[i])
    
    test = val_X[best[i]].reshape(32,32).T
    plt.contourf(xlist/1.889,ylist/1.889,test,)
    
    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])
    
    CS2 = plt.contour(X/1.889, Y/1.889, predictions_val[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='r', linestyles="solid")
    plt.clabel(CS2, inline=1, fontsize=10)
    # plt.show()
    
    CS = plt.contour(X/1.889, Y/1.889, val_y[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', family='Helvetica')
    plt.axes().set_aspect('equal')
    plt.rcParams["figure.figsize"] = (4,4)
    plt.xticks(np.arange(-0.25, 1.75, step=0.5))
    plt.yticks(np.arange(-0.25, 1.75, step=0.5))
    
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
    
    plt.ylabel('Position (Å)')
    plt.xlabel('Position (Å)')  
    # plt.title('Example: ' + str(best[i]))
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.show()
#%%


for i in range(10):
    
    # true = val_y[i].reshape(32,32)
    # plt.contourf(xlist,ylist,true,levels = [-1,0,2,4,6,10,15,20,30])
    # plt.show()

    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,2,4,6,10,15,20,30])
    # plt.show()
    
    # err = true-pred
    # plt.contourf(xlist,ylist,err,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
    # plt.show()

    test = val_X[i].reshape(32,32).T
    plt.contourf(xlist,ylist,test)
    
    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])
    
    CS2 = plt.contour(X, Y, predictions_val[i].reshape(32,32),levels = [-1,0,1,2,4,6,10,15,20,30], colors='r', linestyles="solid")
    plt.clabel(CS2, inline=1, fontsize=10)
    # plt.show()
    
    CS = plt.contour(X, Y, val_y[i].reshape(32,32),levels = [-1,0,1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()
    
    # test = val_X[i].reshape(32,32).T
    # plt.contourf(xlist,ylist,test*627,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
    # plt.colorbar()
    # plt.show()    
    
#%%
    
""" 3D POTENTIAL --> ENERGIES """


#%%

@jit(nopython=True)
def fgh_3D(domain,potential,mass):
    
    n = len(domain)
    dx = domain[1]-domain[0]
    k = np.pi/dx
    
    vmat = 0
    tmat = 0
    
    hmat = np.zeros((n**3,n**3))
    
    for xi in range(n):
        for xj in range(n):
            for yi in range(n):
                for yj in range(n):
                    for zi in range(n):
                        for zj in range(n):
                            if xi == xj and yi == yj and zi == zj:
                                vmat = potential[xj,yj,zj]
                                tmat = (k**2)
                            elif xi != xj and yi == yj and zi == zj:
                                dji = xj-xi
                                vmat = 0
                                tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            elif xi == xj and yi != yj and zi == zj:
                                dji = yj-yi
                                vmat = 0
                                tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            elif xi == xj and yi == yj and zi != zj:
                                dji = zj-zi
                                vmat = 0
                                tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            else:
                                vmat = 0
                                tmat = 0
                            hmat[xi+n*yi+n*n*zi,xj+n*yj+n*n*zj] = (1/(2*mass))*tmat + vmat
    
    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln

#%%

potentials_3D = []

grid = 16

for i in range(500):
    
    # print(i)
    
    xlist = np.linspace(-0.5,1.5,grid)*1.8897
    ylist = np.linspace(-0.5,1.5,grid)*1.8897
    zlist = np.linspace(-0.5,1.5,grid)*1.8897
    
    X, Y, Z = np.meshgrid(xlist,ylist,zlist)
    
    KX_1 = (3000*np.random.random()+1500)/(350*627)
    KY_1 = (3000*np.random.random()+1500)/(350*627)
    KZ_1 = (3000*np.random.random()+1500)/(350*627)
    
    KX_2 = (3000*np.random.random()+1500)/(350*627)
    KY_2 = (3000*np.random.random()+1500)/(350*627)
    KZ_2 = (3000*np.random.random()+1500)/(350*627)
    
    X2 = -2*np.random.random()-0.5
    X3 = -2*np.random.random()-0.5
    X4 = -2*np.random.random()-0.5
    Y3 = -2*np.random.random()-0.5
    Y4 = -2*np.random.random()-0.5
    Z4 = -2*np.random.random()-0.5
    
    dE_2 = 2*np.random.random()/627
    dE_3 = 2*np.random.random()/627 + dE_2
    dE_4 = 2*np.random.random()/627 + dE_3
    
    potential_1 = 0.5*KX_1*(X)**2 + 0.5*KY_1*(Y)**2 + 0.5*KZ_1*(Z)**2
    potential_2 = 0.5*KX_2*(X+X2)**2 + 0.5*KY_2*(Y)**2 + 0.5*KZ_2*(Z)**2 + dE_2
    potential_3 = 0.5*KX_2*(X+X3)**2 + 0.5*KY_2*(Y+Y3)**2 + 0.5*KZ_2*(Z)**2 + dE_3
    potential_4 = 0.5*KX_2*(X+X4)**2 + 0.5*KY_2*(Y+Y4)**2 + 0.5*KZ_2*(Z+Z4)**2 + dE_4
    
    # X2 = -2*np.random.random()-0.5
    # X3 = -2*np.random.random()-0.5
    # Y3 = -2*np.random.random()-0.5
    
    # KX_1 = (3000*np.random.random()+1500)/(350*627)
    # KX_2 = (3000*np.random.random()+1500)/(350*627)
    # KX_3 = (3000*np.random.random()+1500)/(350*627)
    
    # KY_1 = (3000*np.random.random()+1500)/(350*627)
    # KY_2 = (3000*np.random.random()+1500)/(350*627)
    # KY_3 = (3000*np.random.random()+1500)/(350*627)
    
    # dE_2 = 2*np.random.random()/627
    # dE_3 = 2*np.random.random()/627+dE_2
    
    # xlist = np.linspace(-0.5,1.5,grid)*1.8897
    # ylist = np.linspace(-0.5,1.5,grid)*1.8897
    
    # X, Y = np.meshgrid(xlist,ylist)
    
    # potential_1 = 0.5*KX_1*(X)**2 + 0.5*KY_1*Y**2 
    # potential_2 = 0.5*KX_2*(X+X2)**2 + 0.5*KY_2*Y**2 + dE_2
    # potential_3 = 0.5*KX_3*(X+X3)**2 + 0.5*KY_3*(Y+Y3)**2 + dE_3
    
    couplings = np.full((grid,grid,grid),1)*(10/627)
    two_state = np.array([[potential_1,couplings,couplings,couplings],[couplings,potential_2,couplings,couplings],[couplings,couplings,potential_3,couplings],[couplings,couplings,couplings,potential_4]])
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES = ground_state_PES - np.min(ground_state_PES)
    potentials_3D.append(ground_state_PES)
    
    # plt.contourf(xlist,ylist,ground_state_PES*627,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
    # plt.show()

potentials_3D = np.array(potentials_3D)
print(potentials_3D)

#%%

start = time.time()

target_potentials =[]
# target_wavefunctions = []
target_energies = []

for i in range(500):

    target_potentials.append(potentials_3D[i].reshape(grid**3))
    
    temp = fgh_3D(xlist,potentials_3D[i],1836)
    # target_wavefunctions.append(temp[1].T[0:5].T)
    target_energies.append(temp[0][0:50])
    
    print(i)
    
target_potentials = np.array(target_potentials)
# target_wavefunctions = np.array(target_wavefunctions)
target_energies = np.array(target_energies)

end = time.time()
print(end-start)

#%%

print(target_potentials.shape)

#%%

train = target_potentials
target = target_energies*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

NN_model = Sequential()
NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.0001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

for i in range(6):
    NN_model.fit(train_X_tf, train_y_tf, epochs=1024, batch_size=16*2**i, verbose=2)

NN_model.save('/Users/maximsecor/Desktop/ANN_2D/model_3d_pot_nrg')

#%%

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

print(np.min(val_y[:,:5]),np.max(val_y[:,:5]))

#%%

"Error in energies and excitation energies"

print("Energy Errors")
energy_error = predictions_val - val_y

print(np.mean(abs(energy_error)))

for i in range(50):
    print("Energy Error of State "+str(int(i))+": ",np.mean(np.abs(energy_error[:,i])))

# print("IR errors")

# excitations_pred = []
# excitations_true = []

# for i in range(49):
#     pred_IR = predictions_val[:,i+1] - predictions_val[:,0]
#     true_IR = val_y[:,i+1] - val_y[:,0]
#     IR_error = pred_IR - true_IR
#     excitations_pred.append(pred_IR)
#     excitations_true.append(true_IR)
#     print("IR Error of State "+str(int(i+1))+": ",350*np.mean(np.abs(IR_error[i])))
    
# excitations_pred = np.array(excitations_pred)
# excitations_true = np.array(excitations_true)

#%%

"Error Histogram in wavenumbers"

energy_error_flat = energy_error.reshape(energy_error.shape[0]*energy_error.shape[1])
print(energy_error_flat)

plt.xticks(np.arange(-0.5, 0.75, step=0.25))
ticks = np.linspace(-0.5,0.5,100)
plt.hist(energy_error_flat, bins=ticks)
plt.ylabel('Counts(kcal/mol)')
plt.xlabel('ANN Error (kcal/mol)')

plt.show()

#%%

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_val)))

for i in range(50):
    plot0[0] = val_y[:,i]
    plot0[1] = predictions_val[:,i]
    data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
    ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('ANN Energy (kcal/mol)')
ax.set_xlabel('FGH Energy (kcal/mol)')
# ax.set_title('Energy Comaprison')

plt.rc('font', family='Helvetica')
plt.axes().set_aspect('equal')
plt.rcParams["figure.figsize"] = (4,4)
plt.xticks(np.arange(0, 14, step=2))
plt.yticks(np.arange(0, 14, step=2))

x = np.linspace(0,np.max(val_y[:,i])+1,1000)
y = x
plt.plot(x, y, '--k', linewidth=1.5)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()

#%%

        
"""""""""""""""""""""""""""




YOU ARE PASSING INTO OLD STUFF




"""""""""""""""""""""""""""

#%%


train = target_wavefunctions[:,:,4]**2
# target = target_wavefunctions[:,:,3]
target = target_potentials*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)


train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)


# NN_model = Sequential()

# NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))

# for layers in range(3):
#     NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))

# NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

# opt = Adam(learning_rate=(0.001))
# NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

# for i in range(6):
#     NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)
    
overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 100)
model = Sequential()
model.add(Dense(dense_nodes, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(dense_layers):
    model.add(Dense(dense_nodes, kernel_initializer='normal',activation='relu'))
model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
opt = Adam(learning_rate=(0.0001))
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

print("\nModel Configured")
start = time.time()
for i in range(6):
    model.fit(train_X_tf, train_y_tf, epochs=160000, batch_size=16*(2**i), verbose=2, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('\nTraining Time: ',(end-start))

#%%
    
# for i in range(2):
#     NN_model.fit(train_X_tf, train_y_tf, epochs=64, batch_size=512*2**i, verbose=2)

#%%

predictions_val = model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

error = np.mean(np.abs(predictions_val-val_y),1)
best = np.where((error<0.35)==True)[0]
print(best.shape)

#%%

for i in range(len(best)):
    
    print(i)
    
    test = val_X[best[i]].reshape(32,32).T
    plt.contourf(xlist,ylist,test,)
    
    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])
    
    CS2 = plt.contour(X, Y, predictions_val[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='r', linestyles="solid")
    plt.clabel(CS2, inline=1, fontsize=10)
    # plt.show()
    
    CS = plt.contour(X, Y, val_y[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', family='Helvetica')
    plt.axes().set_aspect('equal')
    plt.rcParams["figure.figsize"] = (4,4)
    plt.xticks(np.arange(-0.5, 3, step=1))
    plt.yticks(np.arange(-0.5, 3, step=1))
    
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.show()

#%%



for i in range(10):
    
    # true = val_y[i].reshape(32,32)
    # plt.contourf(xlist,ylist,true,levels = [-1,0,2,4,6,10,15,20,30])
    # plt.show()

    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,2,4,6,10,15,20,30])
    # plt.show()
    
    # err = true-pred
    # plt.contourf(xlist,ylist,err,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
    # plt.show()

    test = val_X[i].reshape(32,32).T
    plt.contourf(xlist,ylist,test)
    
    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])
    
    CS2 = plt.contour(X, Y, predictions_val[i].reshape(32,32),levels = [-1,0,1,2,4,6,10,15,20,30], colors='r', linestyles="solid")
    plt.clabel(CS2, inline=1, fontsize=10)
    # plt.show()
    
    CS = plt.contour(X, Y, val_y[i].reshape(32,32),levels = [-1,0,1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()
    
    # test = val_X[i].reshape(32,32).T
    # plt.contourf(xlist,ylist,test*627,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
    # plt.colorbar()
    # plt.show()
    
#%%
    
train = target_energies*627
target = target_potentials*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

dense_layers = 3
dense_nodes = 512

# NN_model = Sequential()

overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 100)
model = Sequential()
model.add(Dense(dense_nodes, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(dense_layers):
    model.add(Dense(dense_nodes, kernel_initializer='normal',activation='relu'))
model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))
opt = Adam(learning_rate=(0.0001))
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

print("\nModel Configured")
start = time.time()
for i in range(6):
    model.fit(train_X_tf, train_y_tf, epochs=160000, batch_size=16*(2**i), verbose=2, validation_data=(val_X_tf, val_y_tf), callbacks=[overfitCallback])
end = time.time()
print('\nTraining Time: ',(end-start))

# NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))

# for layers in range(3):
#     NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))

# NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

# opt = Adam(learning_rate=(0.001))
# NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

# train_X_tf = tf.convert_to_tensor(train_X, np.float32)
# train_y_tf = tf.convert_to_tensor(train_y, np.float32)
# val_X_tf = tf.convert_to_tensor(val_X, np.float32)
# val_y_tf = tf.convert_to_tensor(val_y, np.float32)

# for i in range(6):
#     NN_model.fit(train_X_tf, train_y_tf, epochs=512, batch_size=16*2**i, verbose=2)

#%%
    
# for i in range(2):
#     NN_model.fit(train_X_tf, train_y_tf, epochs=64, batch_size=512*2**i, verbose=2)

#%%

predictions_val = model.predict(val_X_tf)
MAE = mean_absolute_error(val_y_tf, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

error = np.mean(np.abs(predictions_val-val_y),1)
best = np.where((error<0.5)==True)[0]
print(best)

#%%

for i in range(5):
    predictedY = val_X[i,:10]
    s = np.full(len(predictedY),200)
    
    plt.scatter(np.zeros_like(predictedY),predictedY,s,marker="_")
    plt.rcParams["figure.figsize"] = (0.25,4)
    
    plt.xticks([])
    
    plt.yticks(np.arange(0.0, 10, step=2))
    
    plt.rcParams['axes.facecolor'] = 'w'
    
    # plt.set_marker(self, |)
    plt.show()
    
#%%
    
print(np.full(10,2))

#%%

best = np.array([71,120,160,37,96])

for i in range(len(best)):
# for i in range(5):
    
    print(best[i])
    
    # test = val_X[best[i]].reshape(32,32).T
    # plt.contourf(xlist,ylist,test)
    
    # pred = predictions_val[i].reshape(32,32)
    # plt.contourf(xlist,ylist,pred,levels = [-1,0,1,2,4,6,10,15,20,30])
    
    CS2 = plt.contour(X, Y, predictions_val[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='r', linestyles="solid")
    plt.clabel(CS2, inline=1, fontsize=10)
    # plt.show()
    
    CS = plt.contour(X, Y, val_y[best[i]].reshape(32,32),levels = [1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
    plt.clabel(CS, inline=1, fontsize=10)
    
    # plt.title('Example: ' + str(best[i]))
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', family='Helvetica')
    plt.axes().set_aspect('equal')
    plt.rcParams["figure.figsize"] = (4,4)
    plt.xticks(np.arange(-0.5, 3, step=1))
    plt.yticks(np.arange(-0.5, 3, step=1))
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.rcParams['axes.facecolor'] = 'black'
    
    plt.grid(False)
    # plt.right_ax.grid(False)
    
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
    
    plt.show()
    
    # predictedY = val_X[i,:10]*627
    # plt.scatter(predictedY, np.zeros_like(predictedY))
    # plt.rcParams["figure.figsize"] = (4,0.25)
    # plt.xticks(np.arange(0.0, 10, step=2))
    # plt.yticks([])
    # plt.show()
    
#%%
    
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

#%%

for i in range(len(best)):
# # for i in range(5):
        
    q = best[i]
    predictedY = val_X[q,:10]
    s = np.full(len(predictedY),200)
    
    plt.scatter(np.zeros_like(predictedY),predictedY,s,marker="_")
    plt.rcParams["figure.figsize"] = (0.25,4)
    plt.xticks([])
    plt.yticks(np.arange(0.0, 8, step=2))
    # plt.rcParams['axes.facecolor'] = 'w'
    
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    
    plt.rc('font', family='Helvetica')
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
    
    # plt.set_marker(self, |)
    plt.show()
        
#%%
   
predictedY = val_X[i,:10]
s = np.full(len(predictedY),200)

plt.scatter(np.zeros_like(predictedY),predictedY,s,marker="_")
plt.rcParams["figure.figsize"] = (0.25,4)

plt.xticks([])

plt.yticks(np.arange(0.0, 10, step=2))

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', family='Helvetica')

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rcParams['axes.facecolor'] = 'white'

# plt.set_marker(self, |)
plt.show()

#%%    

q = 135
    
CS2 = plt.contour(X, Y, predictions_val[q].reshape(32,32),levels = [-1,0,1,2,4,6,10,15,20,30], colors='r', linestyles="solid")
plt.clabel(CS2, inline=1, fontsize=10)
# plt.show()

CS = plt.contour(X, Y, val_y[q].reshape(32,32),levels = [-1,0,1,2,4,6,10,15,20,30], colors='w', linestyles="solid")
plt.clabel(CS, inline=1, fontsize=10)

# plt.title('Example: ' + str(best[i]))

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', family='Helvetica')
plt.axes().set_aspect('equal')
plt.rcParams["figure.figsize"] = (4,4)
plt.xticks(np.arange(-0.5, 3, step=1))
plt.yticks(np.arange(-0.5, 3, step=1))

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams['axes.facecolor'] = 'black'

plt.grid(False)
# plt.right_ax.grid(False)

plt.show()

#%%
    
@jit(nopython=True)
def fgh_3D(domain,potential,mass):
    
    n = len(domain)
    dx = domain[1]-domain[0]
    k = np.pi/dx
    
    vmat = 0
    tmat = 0
    
    hmat = np.zeros((n**3,n**3))
    
    for xi in range(n):
        for xj in range(n):
            for yi in range(n):
                for yj in range(n):
                    for zi in range(n):
                        for zj in range(n):
                            if xi == xj and yi == yj and zi == zj:
                                vmat = potential[xj,yj,zj]
                                tmat = (k**2)
                            elif xi != xj and yi == yj and zi == zj:
                                dji = xj-xi
                                vmat = 0
                                tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            elif xi == xj and yi != yj and zi == zj:
                                dji = yj-yi
                                vmat = 0
                                tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            elif xi == xj and yi == yj and zi != zj:
                                dji = zj-zi
                                vmat = 0
                                tmat = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
                            else:
                                vmat = 0
                                tmat = 0
                            hmat[xi+n*yi+n*n*zi,xj+n*yj+n*n*zj] = (1/(2*mass))*tmat + vmat
    
    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln

#%%

   
potentials_3D = []

grid = 16

for i in range(250):
    
    # print(i)
    
    xlist = np.linspace(-0.5,1.5,grid)*1.8897
    ylist = np.linspace(-0.5,1.5,grid)*1.8897
    zlist = np.linspace(-0.5,1.5,grid)*1.8897
    
    X, Y, Z = np.meshgrid(xlist,ylist,zlist)
    
    KX_1 = (3000*np.random.random()+1500)/(350*627)
    KY_1 = (3000*np.random.random()+1500)/(350*627)
    KZ_1 = (3000*np.random.random()+1500)/(350*627)
    
    KX_2 = (3000*np.random.random()+1500)/(350*627)
    KY_2 = (3000*np.random.random()+1500)/(350*627)
    KZ_2 = (3000*np.random.random()+1500)/(350*627)
    
    X2 = -2*np.random.random()-0.5
    X3 = -2*np.random.random()-0.5
    X4 = -2*np.random.random()-0.5
    Y3 = -2*np.random.random()-0.5
    Y4 = -2*np.random.random()-0.5
    Z4 = -2*np.random.random()-0.5
    
    dE_2 = 2*np.random.random()/627
    dE_3 = 2*np.random.random()/627 + dE_2
    dE_4 = 2*np.random.random()/627 + dE_3
    
    potential_1 = 0.5*KX_1*(X)**2 + 0.5*KY_1*(Y)**2 + 0.5*KZ_1*(Z)**2
    potential_2 = 0.5*KX_2*(X+X2)**2 + 0.5*KY_2*(Y)**2 + 0.5*KZ_2*(Z)**2 + dE_2
    potential_3 = 0.5*KX_2*(X+X3)**2 + 0.5*KY_2*(Y+Y3)**2 + 0.5*KZ_2*(Z)**2 + dE_3
    potential_4 = 0.5*KX_2*(X+X4)**2 + 0.5*KY_2*(Y+Y4)**2 + 0.5*KZ_2*(Z+Z4)**2 + dE_4
    
    # X2 = -2*np.random.random()-0.5
    # X3 = -2*np.random.random()-0.5
    # Y3 = -2*np.random.random()-0.5
    
    # KX_1 = (3000*np.random.random()+1500)/(350*627)
    # KX_2 = (3000*np.random.random()+1500)/(350*627)
    # KX_3 = (3000*np.random.random()+1500)/(350*627)
    
    # KY_1 = (3000*np.random.random()+1500)/(350*627)
    # KY_2 = (3000*np.random.random()+1500)/(350*627)
    # KY_3 = (3000*np.random.random()+1500)/(350*627)
    
    # dE_2 = 2*np.random.random()/627
    # dE_3 = 2*np.random.random()/627+dE_2
    
    # xlist = np.linspace(-0.5,1.5,grid)*1.8897
    # ylist = np.linspace(-0.5,1.5,grid)*1.8897
    
    # X, Y = np.meshgrid(xlist,ylist)
    
    # potential_1 = 0.5*KX_1*(X)**2 + 0.5*KY_1*Y**2 
    # potential_2 = 0.5*KX_2*(X+X2)**2 + 0.5*KY_2*Y**2 + dE_2
    # potential_3 = 0.5*KX_3*(X+X3)**2 + 0.5*KY_3*(Y+Y3)**2 + dE_3
    
    couplings = np.full((grid,grid,grid),1)*(10/627)
    two_state = np.array([[potential_1,couplings,couplings,couplings],[couplings,potential_2,couplings,couplings],[couplings,couplings,potential_3,couplings],[couplings,couplings,couplings,potential_4]])
    ground_state_PES = np.linalg.eigh(two_state.T)[0].T[0]
    ground_state_PES = ground_state_PES - np.min(ground_state_PES)
    potentials_3D.append(ground_state_PES)
    
    # plt.contourf(xlist,ylist,ground_state_PES*627,levels = [-1,0,1,2,3,4,5,6,10,15,20,30])
    # plt.show()

potentials_3D = np.array(potentials_3D)
print(potentials_3D)

#%%

start = time.time()

target_potentials =[]
target_wavefunctions = []
target_energies = []

for i in range(250):

    target_potentials.append(potentials_3D[i].reshape(grid**3))
    
    temp = fgh_3D(xlist,potentials_3D[i],1836)
    target_wavefunctions.append(temp[1].T[0:5].T)
    target_energies.append(temp[0][0:50])
    
    print(i)
    
target_potentials = np.array(target_potentials)
target_wavefunctions = np.array(target_wavefunctions)
target_energies = np.array(target_energies)

end = time.time()
print(end-start)

#%%

print(target_potentials.shape)

#%%

train = target_potentials
target = target_energies*627

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.1, random_state = 2021)

NN_model = Sequential()
NN_model.add(Dense(512, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))
for layers in range(3):
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(len(train_y[0]), kernel_initializer='normal',activation='linear'))

opt = Adam(learning_rate=(0.0001))
NN_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)

NN_model.fit(train_X_tf, train_y_tf, epochs=4096, batch_size=32, verbose=2)

#%%

predictions_val = NN_model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error =', MAE)

predictions_train = NN_model.predict(train_X_tf)
MAE = mean_absolute_error(train_y_tf, predictions_train)
print('Training Set Error =', MAE)

#%%

"Error in energies and excitation energies"

print("Energy Errors")
energy_error = predictions_val - val_y

print(np.mean(abs(energy_error)))

for i in range(50):
    print("Energy Error of State "+str(int(i))+": ",np.mean(np.abs(energy_error[:,i])))


print("IR errors")

excitations_pred = []
excitations_true = []

for i in range(49):
    pred_IR = predictions_val[:,i+1] - predictions_val[:,0]
    true_IR = val_y[:,i+1] - val_y[:,0]
    IR_error = pred_IR - true_IR
    excitations_pred.append(pred_IR)
    excitations_true.append(true_IR)
    print("IR Error of State "+str(int(i+1))+": ",350*np.mean(np.abs(IR_error[i])))
    
excitations_pred = np.array(excitations_pred)
excitations_true = np.array(excitations_true)

#%%

"Error Histogram in wavenumbers"

energy_error_flat = energy_error.reshape(energy_error.shape[0]*energy_error.shape[1])
print(energy_error_flat)

ticks = np.linspace(-1,1,100)
plt.hist(energy_error_flat, bins=ticks) 
plt.show()

#%%

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_val)))

for i in range(50):
    plot0[0] = val_y[:,i]
    plot0[1] = predictions_val[:,i]
    data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
    ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('ANN Energy (kcal/mol)')
ax.set_xlabel('FGH Energy (kcal/mol)')
# ax.set_title('Energy Comaprison')

plt.rc('font', family='Helvetica')

# plt.axes().set_aspect('equal')

x = np.linspace(0,np.max(val_y[:,i])+1,1000)
y = x
plt.plot(x, y, '--k', linewidth=1.5)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()

    