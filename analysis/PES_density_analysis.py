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

density_true = []
density_pred = []
true_pos_list = []
pred_pos_list = []

MAE_CV_total = []
track_error_pos = []

for i in range(5):
    
    
    features_potential[features_potential>0.159] = 0.159
    features = features_potential
    
    train_X, test_X, train_y, test_y = train_test_split(features, target_wave[:,:,i]**2, test_size = 0.1, random_state = 1103)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.11111, random_state = 120)
    
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    test_X_tf = tf.convert_to_tensor(test_X, np.float32)
    test_y_tf = tf.convert_to_tensor(test_y, np.float32)

    model = load_model(directory_loc + 'ANN_1DTISE/PES_density/DENS_'+str(int(i))+'/saved_model')

    predictions_train = model.predict(train_X_tf)
    MAE = mean_absolute_error(train_y_tf, predictions_train)
    print('Train Set Error (kcal/mol) =', MAE)

    predictions_val = model.predict(val_X_tf)
    MAE = mean_absolute_error(val_y_tf, predictions_val)
    print('Cross-Validation Set Error (kcal/mol) =', MAE)
    
    predictions_test = model.predict(test_X_tf)
    MAE = mean_absolute_error(test_y_tf, predictions_test)
    print('Test Set Error (kcal/mol) =', MAE)
    
    predictions_val[predictions_test<0] = 0
    norm_fact = 1/np.sum(predictions_test,1)
    predictions_norm = (predictions_test.T * norm_fact).T

    density_true.append(test_y)
    density_pred.append(predictions_norm)
    
    MAE_CV_total.append(MAE)

    # for j in range(10):
    #     plt.ylim(-1,50)
    #     plt.plot(domain,val_y[j]*1000+10,domain,predictions_val[j]*1000+10,domain,val_X[j]*627)
    #     plt.show()
    
    true_pos = np.sum(domain*test_y,1)
    pred_pos = np.sum(domain*predictions_norm,1)
    true_pos_list.append(true_pos)
    pred_pos_list.append(pred_pos)
    
    error_pos = true_pos - pred_pos
    print(np.mean(np.abs(error_pos)))
    track_error_pos.append(np.mean(np.abs(error_pos)))
    
    print('\n')
    
    # plot_pos = np.zeros((2,len(predictions_norm)))
    
    # plot_pos[0] = true_pos
    # plot_pos[1] = pred_pos
    
    # data_values = pd.DataFrame(plot_pos.T,columns=["FGH Calculation", "ML Calculation"])
    # ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
    # ax.set_ylabel('')
    # ax.set_xlabel('')
    # plt.ylim(-0.1,1.0)
    # plt.xlim(-0.1,1.0)
    # plt.show()

    # ticks = np.linspace(-0.5,0.5,40)
    # plt.hist(error_pos, bins=ticks) 
    # plt.show()
    
    # true_pos = np.sum(val_X*627*val_y,1)
    # pred_pos = np.sum(val_X*627*predictions_norm,1)
    
    # error_pos = pred_pos - true_pos
    
    # plot_pos = np.zeros((2,len(predictions_norm)))
    
    # plot_pos[0] = true_pos
    # plot_pos[1] = pred_pos
    # data_values = pd.DataFrame(plot_pos.T,columns=["FGH Calculation", "ML Calculation"])
    # ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
    # ax.set_ylabel('')
    # ax.set_xlabel('')
    # plt.ylim(0,10)
    # plt.xlim(0,10)
    # plt.show()
    
    # ticks = np.linspace(0,5,40)
    # plt.hist(error_pos, bins=ticks) 
    # plt.show()
    
density_true = np.array(density_true)
density_pred = np.array(density_pred)

MAE_CV_total = np.array(MAE_CV_total)
track_error_pos = np.array(track_error_pos)

print(MAE_CV_total)
print(np.mean(MAE_CV_total)/5)


print(track_error_pos)
print(np.mean(track_error_pos)/5)

#%%

train_X, val_X, train_y, val_y = train_test_split(features, target_energies, test_size = 0.1, random_state = 1103)

#%%

print(val_y[0,0])

#%%

for i in range(100):

    add = val_y[i]*627
    
    scale = add[4]+5
    scale_den = scale*20
    plt.ylim(-1,scale)
    
    plt.plot(domain,val_X[i]*627,'k', label='Potential')
    plt.plot(domain,np.abs(density_true[0,i])*scale_den+add[0],'r', label='FGH')
    plt.plot(domain,np.abs(density_pred[0,i])*scale_den+add[0],'b', label='ANN')
    
    plt.plot(domain,density_true[1,i]*scale_den+add[1],'r',domain,density_pred[1,i]*scale_den+add[1],'b')
    plt.plot(domain,density_true[2,i]*scale_den+add[2],'r',domain,density_pred[2,i]*scale_den+add[2],'b')
    plt.plot(domain,density_true[3,i]*scale_den+add[3],'r',domain,density_pred[3,i]*scale_den+add[3],'b')
    plt.plot(domain,density_true[4,i]*scale_den+add[4],'r',domain,density_pred[4,i]*scale_den+add[4],'b')
    
    # plt.legend()
    
    plt.xlabel('Position (Å)')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('Wavefunction Comaprison '+str(int(i)))
    
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
    plt.text(-0.5, add[1]+0.5, r'$\rho_{1}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[2]+0.5, r'$\rho_{2}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[3]+0.5, r'$\rho_{3}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[4]+0.5, r'$\rho_{4}$', fontsize=SMALL_SIZE)
    
    plt.show()
    # plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_sample_10.tif',dpi=1200)
        

#%%
    
i = 0

add = val_y[i]*627
scale = add[4]*1.2
scale_den = scale*15
plt.ylim(-1,scale)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(density_true[0,i])*scale_den+add[0],'r', label='FGH')
plt.plot(domain,np.abs(density_pred[0,i])*scale_den+add[0],'--b', label='ANN')

plt.plot(domain,density_true[1,i]*scale_den+add[1],'r',domain,density_pred[1,i]*scale_den+add[1],'--b')
plt.plot(domain,density_true[2,i]*scale_den+add[2],'r',domain,density_pred[2,i]*scale_den+add[2],'--b')
plt.plot(domain,density_true[3,i]*scale_den+add[3],'r',domain,density_pred[3,i]*scale_den+add[3],'--b')
plt.plot(domain,density_true[4,i]*scale_den+add[4],'r',domain,density_pred[4,i]*scale_den+add[4],'--b')

# plt.legend()

plt.hlines(add[0], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.5, 1.5, colors='k', linewidth=0.5)

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
# plt.title('Density Comaprison: Asymmetric Double Well')

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
plt.text(-0.5, add[1]+0.5, r'$\rho_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.5, r'$\rho_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$\rho_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$\rho_{4}$', fontsize=SMALL_SIZE)

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_density_ADW.tif',dpi=1200)

#%%
    
i = 50

add = val_y[i]*627
scale = add[4]*1.2
scale_den = scale*15
plt.ylim(-1,scale)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(density_true[0,i])*scale_den+add[0],'r', label='FGH')
plt.plot(domain,np.abs(density_pred[0,i])*scale_den+add[0],'--b', label='ANN')

plt.plot(domain,density_true[1,i]*scale_den+add[1],'r',domain,density_pred[1,i]*scale_den+add[1],'--b')
plt.plot(domain,density_true[2,i]*scale_den+add[2],'r',domain,density_pred[2,i]*scale_den+add[2],'--b')
plt.plot(domain,density_true[3,i]*scale_den+add[3],'r',domain,density_pred[3,i]*scale_den+add[3],'--b')
plt.plot(domain,density_true[4,i]*scale_den+add[4],'r',domain,density_pred[4,i]*scale_den+add[4],'--b')

# plt.legend()

plt.hlines(add[0], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.5, 1.5, colors='k', linewidth=0.5)

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
# plt.title('Density Comaprison: Harmonic Oscillator')

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
plt.text(-0.5, add[1]+0.5, r'$\rho_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.5, r'$\rho_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$\rho_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$\rho_{4}$', fontsize=SMALL_SIZE)

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_density_HO.tif',dpi=1200)

#%%
    
i = 65

add = val_y[i]*627
scale = add[4]*1.2
scale_den = scale*15
plt.ylim(-1,scale)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(density_true[0,i])*scale_den+add[0],'r', label='FGH')
plt.plot(domain,np.abs(density_pred[0,i])*scale_den+add[0],'--b', label='ANN')

plt.plot(domain,density_true[1,i]*scale_den+add[1],'r',domain,density_pred[1,i]*scale_den+add[1],'--b')
plt.plot(domain,density_true[2,i]*scale_den+add[2],'r',domain,density_pred[2,i]*scale_den+add[2],'--b')
plt.plot(domain,density_true[3,i]*scale_den+add[3],'r',domain,density_pred[3,i]*scale_den+add[3],'--b')
plt.plot(domain,density_true[4,i]*scale_den+add[4],'r',domain,density_pred[4,i]*scale_den+add[4],'--b')

# plt.legend()

plt.hlines(add[0], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.5, 1.5, colors='k', linewidth=0.5)

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
# plt.title('Density Comaprison: Symmetric Double Well')

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

plt.text(-0.5, add[0]-1, r'$\rho_{0}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[1]+0.5, r'$\rho_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.5, r'$\rho_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$\rho_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$\rho_{4}$', fontsize=SMALL_SIZE)

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_density_SDW.tif',dpi=1200)

#%%
    
i = 66

add = val_y[i]*627
scale = add[4]*1.2
scale_den = scale*15
plt.ylim(-1,scale)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(density_true[0,i])*scale_den+add[0],'r', label='FGH')
plt.plot(domain,np.abs(density_pred[0,i])*scale_den+add[0],'--b', label='ANN')

plt.plot(domain,density_true[1,i]*scale_den+add[1],'r',domain,density_pred[1,i]*scale_den+add[1],'--b')
plt.plot(domain,density_true[2,i]*scale_den+add[2],'r',domain,density_pred[2,i]*scale_den+add[2],'--b')
plt.plot(domain,density_true[3,i]*scale_den+add[3],'r',domain,density_pred[3,i]*scale_den+add[3],'--b')
plt.plot(domain,density_true[4,i]*scale_den+add[4],'r',domain,density_pred[4,i]*scale_den+add[4],'--b')

# plt.legend()

plt.hlines(add[0], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.5, 1.5, colors='k', linewidth=0.5)

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
# plt.title('Density Comaprison: Single Well with Shoulder')

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
plt.text(-0.5, add[1]+0.5, r'$\rho_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.5, r'$\rho_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$\rho_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$\rho_{4}$', fontsize=SMALL_SIZE)

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_density_SWS.tif',dpi=1200)

#%%

true_pos_list = np.array(true_pos_list)
pred_pos_list = np.array(pred_pos_list)

print(np.mean(np.abs(pred_pos_list-true_pos_list)))

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_val)))
plot1 = np.zeros((2,len(predictions_val)))
plot2 = np.zeros((2,len(predictions_val)))
plot3 = np.zeros((2,len(predictions_val)))
plot4 = np.zeros((2,len(predictions_val)))


plot4[0] = true_pos_list[4,:]
plot4[1] = pred_pos_list[4,:]
data_values = pd.DataFrame(plot4.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plot3[0] = true_pos_list[3,:]
plot3[1] = pred_pos_list[3,:]
data_values = pd.DataFrame(plot3.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
plot2[0] = true_pos_list[2,:]
plot2[1] = pred_pos_list[2,:]
data_values = pd.DataFrame(plot2.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plot1[0] = true_pos_list[1,:]
plot1[1] = pred_pos_list[1,:]
data_values = pd.DataFrame(plot1.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
plot0[0] = true_pos_list[0,:]
plot0[1] = pred_pos_list[0,:]
data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('ANN Expectation (Å)')
ax.set_xlabel('FGH Expectation (Å)')
# ax.set_title('Position Expectation Value Comaprison')

x = np.linspace(0,1.2,1000)
y = x
plt.plot(x, y, '--k', linewidth=1.5)

# plt.legend()
plt.rc('font', family='Helvetica')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_density_position_expectation.tif',dpi=1200)

#%%

model_0 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_0/saved_model', custom_objects={'custom_loss':custom_loss})
# model_1 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_1/saved_model', custom_objects={'custom_loss':custom_loss})
# model_2 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_2/saved_model', custom_objects={'custom_loss':custom_loss})
# model_3 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_3/saved_model', custom_objects={'custom_loss':custom_loss})
# model_4 = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_4/saved_model', custom_objects={'custom_loss':custom_loss})

#%%

predictions_val = model.predict(val_X_tf)
MAE = mean_absolute_error(val_y, predictions_val)
print('Cross-Validation Set Error (kcal/mol) =', MAE)

predictions_train = model.predict(train_tf)
MAE = mean_absolute_error(target_tf, predictions_train)
print('Training Set Error (kcal/mol) =', MAE)

#%%


predictions_0 = model_0.predict(val_X_tf)
MAE = mean_absolute_error(target_0**2, predictions_0**2)
print('Cross-Validation Set Error (kcal/mol) =', MAE)

predictions_1 = model_1.predict(train_tf)
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










