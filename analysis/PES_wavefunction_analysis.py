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

directory_loc = '/Users/maximsecor/Desktop/Quantum Mechanics Machine Learning/'

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

states_true = []
states_pred = []
true_pos_list = []
pred_pos_list = []
true_pot_list = []
pred_pot_list = []

overlap_err = []

for i in range(5):
    
    # i = i+1
    
    features_potential[features_potential>0.159] = 0.159
    features = features_potential
    
    train_X, test_X, train_y, test_y = train_test_split(features, target_wave[:,:,i], test_size = 0.1, random_state = 1103)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.11111, random_state = 120)
    
    train_X_tf = tf.convert_to_tensor(train_X, np.float32)
    train_y_tf = tf.convert_to_tensor(train_y, np.float32)
    val_X_tf = tf.convert_to_tensor(val_X, np.float32)
    val_y_tf = tf.convert_to_tensor(val_y, np.float32)
    test_X_tf = tf.convert_to_tensor(test_X, np.float32)
    test_y_tf = tf.convert_to_tensor(test_y, np.float32)

    model = load_model(directory_loc + 'ANN_1DTISE/PES_wavefunction/STATE_'+str(int(i))+'/saved_model', custom_objects={'custom_loss':custom_loss})
    
    predictions_train = model.predict(train_X_tf)
    MAE = mean_absolute_error(train_y_tf**2, predictions_train**2)
    print('Train Set Error (kcal/mol) =', MAE)

    predictions_val = model.predict(val_X_tf)
    MAE = mean_absolute_error(val_y_tf**2, predictions_val**2)
    print('Cross-Validation Set Error (kcal/mol) =', MAE)
    
    predictions_test = model.predict(test_X_tf)
    MAE = mean_absolute_error(test_y_tf**2, predictions_test**2)
    print('Test Set Error (kcal/mol) =', MAE)
    
    norm_fact = np.sqrt(1/np.sum(predictions_train**2,1))
    predictions_norm = (predictions_train.T * norm_fact).T
    overlap_error = np.abs(np.sum(predictions_train*train_y,1))
    print(np.mean(overlap_error))

    norm_fact = np.sqrt(1/np.sum(predictions_val**2,1))
    predictions_norm = (predictions_val.T * norm_fact).T
    overlap_error = np.abs(np.sum(predictions_val*val_y,1))
    print(np.mean(overlap_error))

    norm_fact = np.sqrt(1/np.sum(predictions_test**2,1))
    predictions_norm = (predictions_test.T * norm_fact).T
    overlap_error = np.abs(np.sum(predictions_norm*test_y,1))
    print(np.mean(overlap_error))
    
    states_pred.append(predictions_norm)
    states_true.append(test_y)
    
    overlap_err.append(np.mean(overlap_error))
    
    # for j in range(10):
    #     plt.ylim(-1,50)
    #     plt.plot(domain,val_y[j]*100+10,domain,predictions_norm[j]*100+10,domain,val_X[j]*627)
        # plt.show()

    # print("Bad Wavefunctions")
    # print(np.where(np.abs(1-np.abs(np.sum(val_y*predictions_norm,1)))>0.005))
    # print("Excellent Wavefunctions")
    # print(np.where(np.abs(1-np.abs(np.sum(val_y*predictions_norm,1)))<0.00005))
    
    # true_pos = np.sum(domain*val_y**2,1)
    # pred_pos = np.sum(domain*predictions_norm**2,1)
    # error_pos = true_pos - pred_pos
    
    # plot_pos = np.zeros((2,len(predictions_norm)))
    
    # plot_pos[0] = true_pos
    # plot_pos[1] = pred_pos
    # true_pos_list.append(true_pos)
    # pred_pos_list.append(pred_pos)
    
    # data_values = pd.DataFrame(plot_pos.T,columns=["FGH Calculation", "ML Calculation"])
    # ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
    # ax.set_ylabel('')
    # ax.set_xlabel('')
    # plt.ylim(-0.1,1.0)
    # plt.xlim(-0.1,1.0)
    # plt.show()

    # ticks = np.linspace(-0.01,0.01,20)
    # plt.hist(error_pos, bins=ticks) 
    # plt.show()
    
    # true_pot = np.sum(val_X*val_y**2,1)
    # pred_pot = np.sum(val_X*predictions_norm**2,1)
    # error_pot = true_pot - pred_pot
    
    # plot_pot = np.zeros((2,len(predictions_norm)))
    
    # plot_pot[0] = true_pot
    # plot_pot[1] = pred_pot
    # true_pot_list.append(true_pot)
    # pred_pot_list.append(pred_pot)
    
    # data_values = pd.DataFrame(plot_pot.T,columns=["FGH Calculation", "ML Calculation"])
    # ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
    # ax.set_ylabel('')
    # ax.set_xlabel('')
    # plt.ylim(-0.1,1.0)
    # plt.xlim(-0.1,1.0)
    # plt.show()

    # ticks = np.linspace(-0.01,0.01,20)
    # plt.hist(error_pot, bins=ticks) 
    # plt.show()

    # overlap_error = np.abs(np.sum(predictions_norm*test_y,1))
    # print(np.mean(overlap_error))
    # ticks = np.linspace(0.99,1.0,40)
    # plt.hist(overlap_error, bins=ticks)
    # plt.show()

#%%
    
plt.plot(features[0])
predictions_val = model.predict(val_X_tf)
plt.plot(predictions_val[0])

#%%
predictions_val = model.predict(val_X_tf)
plt.plot(domain,val_X_tf[0],domain,predictions_val[0])
plt.show()

predictions_val = model.predict(val_X_tf-0.159)
plt.plot(domain,val_X_tf[0]-0.159,domain,predictions_val[0])
plt.show()
#%%
    
print(np.mean(overlap_err))

#%%
    
states_pred = np.array(states_pred)
print(states_pred.shape)    

states_true = np.array(states_true)
print(states_true.shape)

#%%

train_X, val_X, train_y, val_y = train_test_split(features, target_energies, test_size = 0.1, random_state = 1103)
# train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.11111, random_state = 120)

#%%

print(val_y[0,0])

#%%

temp_2 = []

for i in range(1000):

    # # i = 90
    # add = val_y[i]*627
    
    # plt.ylim(-1,add[4]+10)
    
    # plt.plot(domain,val_X[i]*627,'k', label='Potential')
    # plt.plot(domain,np.abs(states_true[0,i])*15+add[0],'r', label='FGH')
    # plt.plot(domain,np.abs(states_pred[0,i])*15+add[0],'b', label='ANN')
    
    # plt.plot(domain,1*states_true[1,i]*15+add[1],'r',domain,1*states_pred[1,i]*15+add[1],'b')
    # plt.plot(domain,1*states_true[2,i]*15+add[2],'r',domain,1*states_pred[2,i]*15+add[2],'b')
    # plt.plot(domain,1*states_true[3,i]*15+add[3],'r',domain,1*states_pred[3,i]*15+add[3],'b')
    # plt.plot(domain,1*states_true[4,i]*15+add[4],'r',domain,1*states_pred[4,i]*15+add[4],'b')
    
    # # plt.legend()
    
    # plt.xlabel('Position (Å)')
    # plt.ylabel('Energy (kcal/mol)')
    # plt.title('Wavefunction Comaprison '+str(int(i)))
    
    # plt.rc('font', family='Helvetica')
    
    # SMALL_SIZE = 10
    # MEDIUM_SIZE = 12
    # BIGGER_SIZE = 14
    
    # plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # plt.text(-0.5, add[0]+0.5, r'$\Psi_{0}$', fontsize=SMALL_SIZE)
    # plt.text(-0.5, add[1]+0.5, r'$\Psi_{1}$', fontsize=SMALL_SIZE)
    # plt.text(-0.5, add[2]+0.5, r'$\Psi_{2}$', fontsize=SMALL_SIZE)
    # plt.text(-0.5, add[3]+0.5, r'$\Psi_{3}$', fontsize=SMALL_SIZE)
    # plt.text(-0.5, add[4]+0.5, r'$\Psi_{4}$', fontsize=SMALL_SIZE)
    
    # plt.show()
    # # plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_sample_10.tif',dpi=1200)
    
    temp = []
    temp.append(np.abs(np.sum(states_true[0,i]*states_pred[0,i])))
    temp.append(np.abs(np.sum(states_true[1,i]*states_pred[1,i])))
    temp.append(np.abs(np.sum(states_true[2,i]*states_pred[2,i])))
    temp.append(np.abs(np.sum(states_true[3,i]*states_pred[3,i])))
    temp.append(np.abs(np.sum(states_true[4,i]*states_pred[4,i])))
    temp = np.array(temp)
    
    
    print(np.mean(temp))
    
    temp_2.append(np.mean(temp))
    
#%%
    
temp_2 = np.array(temp_2)
best = np.where(temp_2>0.9995)[0]
print(best)

#%%

print(val_y.shape)

#%%

for q in range(len(best)):

    # i = 90
    
    i = best[q]
    
    add = val_y[i]*627
    scale = add[4]*1.2
    scale_wave = scale*0.75
    
    plt.ylim(-1,scale)
    
    plt.plot(domain,val_X[i]*627,'k', label='Potential')
    plt.plot(domain,np.abs(states_true[0,i])*scale_wave+add[0],'r', label='FGH')
    plt.plot(domain,np.abs(states_pred[0,i])*scale_wave+add[0],'--b', label='ANN')
    
    plt.plot(domain,-1*states_true[1,i]*scale_wave+add[1],'r',domain,1*states_pred[1,i]*scale_wave+add[1],'--b')
    plt.plot(domain,1*states_true[2,i]*scale_wave+add[2],'r',domain,1*states_pred[2,i]*scale_wave+add[2],'--b')
    plt.plot(domain,-1*states_true[3,i]*scale_wave+add[3],'r',domain,1*states_pred[3,i]*scale_wave+add[3],'--b')
    plt.plot(domain,-1*states_true[4,i]*scale_wave+add[4],'r',domain,1*states_pred[4,i]*scale_wave+add[4],'--b')
    
    # plt.legend()
    
    plt.hlines(add[0], -0.5, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[1], -0.5, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[2], -0.5, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[3], -0.5, 1.5, colors='k', linewidth=0.5)
    plt.hlines(add[4], -0.5, 1.5, colors='k', linewidth=0.5)
    
    plt.xlabel('Position (Å)')
    plt.ylabel('Energy (kcal/mol)')
    # plt.title('Wavefunction Comaprison: Asymmetric Double Well')
    
    plt.rc('font', family='Helvetica')
    
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
    
    plt.text(-0.5, add[0]-1.5, r'$\Psi_{0}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[1]+0.5, r'$\Psi_{1}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[2]-1.5, r'$\Psi_{2}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[3]+0.5, r'$\Psi_{3}$', fontsize=SMALL_SIZE)
    plt.text(-0.5, add[4]+0.5, r'$\Psi_{4}$', fontsize=SMALL_SIZE)
    
    plt.show()
    # plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_sample_10.tif',dpi=1200)


#%%
    
i = 65
add = val_y[i]*627
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(states_true[0,i])*scale_wave+add[0],'r', label='FGH')
plt.plot(domain,np.abs(states_pred[0,i])*scale_wave+add[0],'--b', label='ANN')

plt.plot(domain,-1*states_true[1,i]*scale_wave+add[1],'r',domain,1*states_pred[1,i]*scale_wave+add[1],'--b')
plt.plot(domain,1*states_true[2,i]*scale_wave+add[2],'r',domain,1*states_pred[2,i]*scale_wave+add[2],'--b')
plt.plot(domain,-1*states_true[3,i]*scale_wave+add[3],'r',domain,1*states_pred[3,i]*scale_wave+add[3],'--b')
plt.plot(domain,-1*states_true[4,i]*scale_wave+add[4],'r',domain,1*states_pred[4,i]*scale_wave+add[4],'--b')

# plt.legend()

plt.hlines(add[0], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.5, 1.5, colors='k', linewidth=0.5)

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
# plt.title('Wavefunction Comaprison: Asymmetric Double Well')

plt.rc('font', family='Helvetica')

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

plt.text(-0.5, add[0]-1.5, r'$\Psi_{0}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[1]+0.5, r'$\Psi_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]-1.5, r'$\Psi_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$\Psi_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$\Psi_{4}$', fontsize=SMALL_SIZE)

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_wavefunctions_SDW.tif',dpi=1200)

#%%
    
i = 68
add = val_y[i]*627
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(states_true[0,i])*scale_wave+add[0],'r', label='FGH')
plt.plot(domain,np.abs(states_pred[0,i])*scale_wave+add[0],'--b', label='ANN')

plt.plot(domain,-1*states_true[1,i]*scale_wave+add[1],'r',domain,1*states_pred[1,i]*scale_wave+add[1],'--b')
plt.plot(domain,-1*states_true[2,i]*scale_wave+add[2],'r',domain,1*states_pred[2,i]*scale_wave+add[2],'--b')
plt.plot(domain,-1*states_true[3,i]*scale_wave+add[3],'r',domain,1*states_pred[3,i]*scale_wave+add[3],'--b')
plt.plot(domain,-1*states_true[4,i]*scale_wave+add[4],'r',domain,1*states_pred[4,i]*scale_wave+add[4],'--b')

# plt.legend()

plt.hlines(add[0], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.5, 1.5, colors='k', linewidth=0.5)

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
# plt.title('Wavefunction Comaprison: Harmonic Oscillator')

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

plt.text(-0.5, add[0]+0.5, r'$\Psi_{0}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[1]+0.5, r'$\Psi_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.5, r'$\Psi_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$\Psi_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$\Psi_{4}$', fontsize=SMALL_SIZE)

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_wavefunctions_ADW.tif',dpi=1200)

#%%
    
i = 101
add = val_y[i]*627
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(states_true[0,i])*scale_wave+add[0],'r', label='FGH')
plt.plot(domain,np.abs(states_pred[0,i])*scale_wave+add[0],'--b', label='ANN')

plt.plot(domain,-1*states_true[1,i]*scale_wave+add[1],'r',domain,1*states_pred[1,i]*scale_wave+add[1],'--b')
plt.plot(domain,-1*states_true[2,i]*scale_wave+add[2],'r',domain,1*states_pred[2,i]*scale_wave+add[2],'--b')
plt.plot(domain,-1*states_true[3,i]*scale_wave+add[3],'r',domain,1*states_pred[3,i]*scale_wave+add[3],'--b')
plt.plot(domain,-1*states_true[4,i]*scale_wave+add[4],'r',domain,1*states_pred[4,i]*scale_wave+add[4],'--b')

# plt.legend()

plt.hlines(add[0], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.5, 1.5, colors='k', linewidth=0.5)

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
# plt.title('Wavefunction Comaprison: Symmetric Double Well')

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

plt.text(-0.5, add[0]+0.5, r'$\Psi_{0}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[1]+0.5, r'$\Psi_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.5, r'$\Psi_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$\Psi_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$\Psi_{4}$', fontsize=SMALL_SIZE)

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_wavefunctions_ASW.tif',dpi=1200)

#%%
    
i = 66
add = val_y[i]*627
scale = add[4]*1.2
scale_wave = scale*0.75

plt.ylim(-1,scale)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(states_true[0,i])*scale_wave+add[0],'r', label='FGH')
plt.plot(domain,np.abs(states_pred[0,i])*scale_wave+add[0],'--b', label='ANN')

plt.plot(domain,1*states_true[1,i]*scale_wave+add[1],'r',domain,1*states_pred[1,i]*scale_wave+add[1],'--b')
plt.plot(domain,1*states_true[2,i]*scale_wave+add[2],'r',domain,-1*states_pred[2,i]*scale_wave+add[2],'--b')
plt.plot(domain,1*states_true[3,i]*scale_wave+add[3],'r',domain,1*states_pred[3,i]*scale_wave+add[3],'--b')
plt.plot(domain,1*states_true[4,i]*scale_wave+add[4],'r',domain,-1*states_pred[4,i]*scale_wave+add[4],'--b')

plt.hlines(add[0], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[1], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[2], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[3], -0.5, 1.5, colors='k', linewidth=0.5)
plt.hlines(add[4], -0.5, 1.5, colors='k', linewidth=0.5)

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
# plt.title('Wavefunction Comaprison: Single Well with Shoulder')

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

plt.text(-0.5, add[0]+0.5, r'$\Psi_{0}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[1]+0.5, r'$\Psi_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.5, r'$\Psi_{2}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[3]+0.5, r'$\Psi_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.5, r'$\Psi_{4}$', fontsize=SMALL_SIZE)

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images_2/Pot_wavefunctions_ASW.tif',dpi=1200)

#%%

true_pos_list = np.array(true_pos_list)
pred_pos_list = np.array(pred_pos_list)

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

# plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_position_expectation.tif',dpi=1200)

#%%

true_pot_list = np.array(true_pot_list)*627
pred_pot_list = np.array(pred_pot_list)*627

"Eigenenergy Error Correlation Scatter Plots"

plot0 = np.zeros((2,len(predictions_val)))
plot1 = np.zeros((2,len(predictions_val)))
plot2 = np.zeros((2,len(predictions_val)))
plot3 = np.zeros((2,len(predictions_val)))
plot4 = np.zeros((2,len(predictions_val)))


plot4[0] = true_pot_list[4,:]
plot4[1] = pred_pot_list[4,:]
data_values = pd.DataFrame(plot4.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values,label=r'$\Psi_{4}$')
ax.set_ylabel('')
ax.set_xlabel('')
plot3[0] = true_pot_list[3,:]
plot3[1] = pred_pot_list[3,:]
data_values = pd.DataFrame(plot3.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values,label=r'$\Psi_{3}$')
plot2[0] = true_pot_list[2,:]
plot2[1] = pred_pot_list[2,:]
data_values = pd.DataFrame(plot2.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values,label=r'$\Psi_{2}$')
ax.set_ylabel('')
ax.set_xlabel('')
plot1[0] = true_pot_list[1,:]
plot1[1] = pred_pot_list[1,:]
data_values = pd.DataFrame(plot1.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values,label=r'$\Psi_{1}$')
plot0[0] = true_pot_list[0,:]
plot0[1] = pred_pot_list[0,:]
data_values = pd.DataFrame(plot0.T,columns=["FGH Calculation", "ML Calculation"])
ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values,label=r'$\Psi_{0}$')
ax.set_ylabel('ANN Expectation (kcal/mol)')
ax.set_xlabel('FGH Expectation (kcal/mol)')
ax.set_title('Potential Energy Expectation Value Comaprison')

plt.legend()
plt.rc('font', family='Helvetica')
# plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_potential_expectation.tif',dpi=1200)


#%%

true_pos_list = np.array(true_pos_list)
true_pos_list_flat = true_pos_list.reshape(5000)
pred_pos_list = np.array(pred_pos_list)
pred_pos_list_flat = pred_pos_list.reshape(5000)

plot_pos = np.zeros((2,5*len(predictions_norm)))

plot_pos[0] = true_pos_list_flat
plot_pos[1] = pred_pos_list_flat

data_values = pd.DataFrame(plot_pos.T,columns=["FGH Calculation", "ML Calculation"])

ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
plt.ylim(-0.1,1.3)
plt.xlim(-0.1,1.3)
plt.show()

pos_error_flat = pred_pos_list_flat - true_pos_list_flat
print(np.mod(np.where(np.abs(pos_error_flat)>0.5),1000)[0])
print("MAE: ", np.mean(np.abs(pos_error_flat)))

#%%

ticks = np.linspace(-0.05,0.05,50)
plt.hist(pos_error_flat, bins=ticks) 

plt.xlabel('Position (Å)')
plt.ylabel('Frequency')
plt.title('Position Expectation Error')

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

# plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_posexperrhist.tif',dpi=1200)

#%%
    
i = 976
add = val_y[i]*627
scale = add[4]+5
scale_wave = scale*0.75

plt.ylim(add[1]-5,add[2]+5)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(states_true[0,i])*scale_wave+add[0],'r', label='FGH')
plt.plot(domain,np.abs(states_pred[0,i])*scale_wave+add[0],'b', label='ANN')

plt.plot(domain,1*states_true[1,i]*30+add[1],'r',domain,1*states_pred[1,i]*30+add[1],'b')
plt.plot(domain,1*states_true[2,i]*30+add[2],'r',domain,-1*states_pred[2,i]*30+add[2],'b')
plt.plot(domain,1*states_true[3,i]*scale_wave+add[3],'r',domain,1*states_pred[3,i]*scale_wave+add[3],'b')
plt.plot(domain,1*states_true[4,i]*scale_wave+add[4],'r',domain,-1*states_pred[4,i]*scale_wave+add[4],'b')

# plt.legend()

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
plt.title('Wavefunction Comaprison: Delocalization Error')

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

# plt.text(-0.5, add[0]+0.5, r'$\Psi_{0}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[1]-0.65, r'$\Psi_{1}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[2]+0.25, r'$\Psi_{2}$', fontsize=SMALL_SIZE)
# plt.text(-0.5, add[3]+0.5, r'$\Psi_{3}$', fontsize=SMALL_SIZE)
# plt.text(-0.5, add[4]+0.5, r'$\Psi_{4}$', fontsize=SMALL_SIZE)

# plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_sample_976.tif',dpi=1200)

#%%
    
i = 732
add = val_y[i]*627
scale = add[4]+5
scale_wave = scale*0.75

plt.ylim(add[4]-5,add[4]+5)

plt.plot(domain,val_X[i]*627,'k', label='Potential')
plt.plot(domain,np.abs(states_true[0,i])*scale_wave+add[0],'r', label='FGH')
plt.plot(domain,np.abs(states_pred[0,i])*scale_wave+add[0],'b', label='ANN')

plt.plot(domain,1*states_true[1,i]*scale_wave+add[1],'r',domain,1*states_pred[1,i]*scale_wave+add[1],'b')
plt.plot(domain,1*states_true[2,i]*scale_wave+add[2],'r',domain,-1*states_pred[2,i]*scale_wave+add[2],'b')
plt.plot(domain,1*states_true[3,i]*scale_wave+add[3],'r',domain,1*states_pred[3,i]*scale_wave+add[3],'b')
plt.plot(domain,1*states_true[4,i]*30+add[4],'r',domain,-1*states_pred[4,i]*30+add[4],'b')

# plt.legend()

plt.xlabel('Position (Å)')
plt.ylabel('Energy (kcal/mol)')
plt.title('Wavefunction Comaprison: Localization Error')

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

# plt.text(-0.5, add[0]+0.5, r'$\Psi_{0}$', fontsize=SMALL_SIZE)
# plt.text(-0.5, add[1]-0.65, r'$\Psi_{1}$', fontsize=SMALL_SIZE)
# plt.text(-0.5, add[2]+0.25, r'$\Psi_{2}$', fontsize=SMALL_SIZE)
# plt.text(-0.5, add[3]+0.5, r'$\Psi_{3}$', fontsize=SMALL_SIZE)
plt.text(-0.5, add[4]+0.25, r'$\Psi_{4}$', fontsize=SMALL_SIZE)

# plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_sample_732.tif',dpi=1200)

#%%

true_pot_list = np.array(true_pot_list)
true_pot_list_flat = true_pot_list.reshape(5000)
pred_pot_list = np.array(pred_pot_list)
pred_pot_list_flat = pred_pot_list.reshape(5000)

plot_pot = np.zeros((2,5*len(predictions_norm)))

plot_pot[0] = true_pot_list_flat
plot_pot[1] = pred_pot_list_flat

data_values = pd.DataFrame(plot_pot.T,columns=["FGH Calculation", "ML Calculation"])

ax = sns.scatterplot(x="ML Calculation", y="FGH Calculation", data=data_values)
ax.set_ylabel('')
ax.set_xlabel('')
# plt.ylim(-0.1,1.3)
# plt.xlim(-0.1,1.3)
plt.show()

pot_error_flat = pred_pot_list_flat - true_pot_list_flat
print(np.mod(np.where(np.abs(pot_error_flat)>0.5),1000)[0])
print("MAE: ", np.mean(np.abs(pot_error_flat)))

#%%

ticks = np.linspace(-0.5,0.5,50)
plt.hist(pot_error_flat, bins=ticks) 

plt.xlabel('Potential Energy (kcal/mol)')
plt.ylabel('Frequency')
plt.title('Potential Energy Expectation Error')

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

# plt.show()
plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_potexperrhist.tif',dpi=1200)

#%%

states_true = np.array(states_true)
states_true_flat = states_true.reshape(5000,1024)

states_pred = np.array(states_pred)
states_pred_flat = states_pred.reshape(5000,1024)

overlaps = np.abs(np.sum(states_pred_flat[:1000]*states_true_flat[:1000],1))
print(np.mean(overlaps))
# print(np.argmin(overlaps[:1000]))
# print(np.min(overlaps[:1000]))
ticks = np.linspace(0.995,1.0,40)
plt.hist(overlaps, bins=ticks)
# plt.show()

plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.title('Overlap Analysis')

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

# plt.show()
# plt.savefig(directory_loc + 'ANN_1DTISE/Images/Pot_wavefunctions_overlaphist.tif',dpi=1200)























