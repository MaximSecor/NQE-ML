#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep  8 08:47:44 2020
@author: maximsecor

##########################################
GENERATE POTENTIALS, ENERGIES, WAVEFUNCTIONS
##########################################
"""

#%%

import numpy as np
from numba import jit

import time
import os
import pandas as pd

from scipy.linalg import expm, sinm, cosm
from scipy.signal import argrelextrema
from scipy.interpolate import BSpline

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

#%%

# FGH function, its faster jitted. Remove jit if you dont want it,
@jit(nopython=True)
def fgh(domain,potential,mass):
    
    nx = len(domain)
    dx = domain[1]-domain[0]
    k = np.pi/dx
    
    vmat = np.zeros((nx,nx))
    tmat = np.zeros((nx,nx))
    hmat = np.zeros((nx,nx))
    
    for i in range(nx):
        for j in range(nx):
            if i == j:
                vmat[i,j] = potential[j]
                tmat[i,j] = (k**2)/3
            else:
                dji = j-i
                vmat[i,j] = 0
                tmat[i,j] = (2*k**2)/(np.pi**2)*(((-1)**dji)/(dji**2))
            hmat[i,j] = (1/(2*mass))*tmat[i,j] + vmat[i,j]
    
    hmat_soln = np.linalg.eigh(hmat)
    return hmat_soln

#%%

########################################################################
########################### Aim (1) Step (1) ###########################
########################################################################

Mioy_potentials = np.zeros((110,1024))

for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP_potentials/d2" + str(idx) + "337_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)    
    data = y.values
    Mioy_potentials[i] = data[:,1]-np.min(data[:,1])
    
for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP_potentials/d2" + str(idx) + "337_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)
    data = y.values    
    Mioy_potentials[11+i] = data[:,2]-np.min(data[:,2])

for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP-CONEt2_Potentials/d2" + str(idx) + "742_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)    
    data = y.values
    Mioy_potentials[22+i] = data[:,1]-np.min(data[:,1])
    
for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP-CONEt2_Potentials/d2" + str(idx) + "742_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)
    data = y.values    
    Mioy_potentials[33+i] = data[:,2]-np.min(data[:,2])
    
for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP-CONH2_Potentials/d2" + str(idx) + "079_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)    
    data = y.values
    Mioy_potentials[44+i] = data[:,1]-np.min(data[:,1])
    
for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP-CONH2_Potentials/d2" + str(idx) + "079_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)
    data = y.values    
    Mioy_potentials[55+i] = data[:,2]-np.min(data[:,2])
    
for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP-COOMe_Potentials/d2" + str(idx) + "841_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)    
    data = y.values
    Mioy_potentials[66+i] = data[:,1]-np.min(data[:,1])
    
for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP-COOMe_Potentials/d2" + str(idx) + "841_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)
    data = y.values    
    Mioy_potentials[77+i] = data[:,2]-np.min(data[:,2])

for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP-COOH_Potentials/d2" + str(idx) + "957_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)    
    data = y.values
    Mioy_potentials[88+i] = data[:,1]-np.min(data[:,1])
    
for i in range(11):
    
    idx = 37 + 5*i
    file = "/Users/maximsecor/Desktop/ANN_1DTISE/BIP_DATA/BIP-COOH_Potentials/d2" + str(idx) + "957_pes.dat"
    y = pd.read_csv(file, delimiter='       ', header=None)
    data = y.values    
    Mioy_potentials[99+i] = data[:,2]-np.min(data[:,2])
   
#%%

domain = np.linspace(-1,1,1024)
Mioy_potentials_aug = Mioy_potentials
Mioy_potentials_aug = np.concatenate((Mioy_potentials_aug,Mioy_potentials_aug[:,::-1]))
test = []
n_sample = 100

for i in range(n_sample):
    
    pot = np.random.randint(0,len(Mioy_potentials_aug),2)
    a = np.random.random(2)
    norm = np.sqrt(np.sum(a*a))
    a = a/norm
    randpotential = a[0]*Mioy_potentials_aug[pot[0]]+a[1]*Mioy_potentials_aug[pot[1]]
    
    a = 20*np.random.random()-10
    b = np.random.random()
    x = np.linspace(-1, 1, 1024)
    bias = (1/(1 + np.exp(-x/0.1))-0.5)*a
    randpotential = randpotential + bias*(1/627)

    a = 100*np.random.random()
    bias = np.linspace(-1, 1, 1024)**2
    randpotential = randpotential + bias*a*(1/627)
    
    a = 2*np.random.random()+0.9
    randpotential = randpotential*a
    
    randpotential = randpotential - np.min(randpotential)
    test.append(randpotential)


#%%

for i in range(50):

    coupling = (10/627)*np.full((1024), 1)
    temp_pot = np.array(test[i])
    pot_1 = temp_pot
    pot_2 = temp_pot[::-1]
    
    temp_1 = np.array([[pot_1,coupling],[coupling,pot_2]])
    temp_2 = np.linalg.eigh(temp_1.T)[0]
    
    gs = temp_2[:,0] - np.min(temp_2[:,0])
    test.append(gs)

#%%

Mioy_potentials_aug = np.array(test)
potentials_referenced = []

for i in range(len(Mioy_potentials_aug)):
    x0 = np.linspace(-1,1,1024)
    y0 = Mioy_potentials_aug[i]
    
    maxInd = argrelextrema(y0, np.less)

    slope = np.gradient(np.gradient(np.gradient(y0)))
    
    if len(maxInd[0])>1:
        
        if y0[maxInd][0] > y0[maxInd][1]:
            y0 = y0[::-1]
            xmin = -1*x0[maxInd][0]
        else:
            xmin = x0[maxInd][0]
        
    else:
        if slope[maxInd[0]+int(0)] > 0 :
            y0 = y0[::-1]
            xmin = -1*x0[maxInd]
        else:
            xmin = x0[maxInd]
    
    spl = BSpline(x0,y0,1)
    y1 = spl(x0)
    maxInd = np.argmin(y1)
    
    x1 = np.linspace(-0.5,1.5,1024)
    y1 = spl(x1+x0[maxInd])
    
    potentials_referenced.append(y1)

#%%

x1 = np.linspace(-0.5,1.5,1024)
dom = 1.88973*x1

for i in range(50):

    a = 2500*np.random.random()+1500
    pot = (0.5*1836*((a/(627*350))**2))*(dom**2)
    potentials_referenced.append(pot)
    
potentials_referenced = np.array(potentials_referenced)

#%%

###################################################################
########################### Solving FGH ###########################
###################################################################

start = time.time()

mass = 1836

features_potential = []
target_energies = []
target_wave = []

n_sample = len(potentials_referenced)

for q in range(100):
    for i in range(int(n_sample/100)):
        
        potential = potentials_referenced[i+q*(int(n_sample/100))]
        features_potential.append(potential)
        
        pr_solutions = fgh(x1*1.88973,potential,mass)
        
        target_energies_temp = pr_solutions[0][0:5]
        target_energies.append(target_energies_temp)
        
        target_wave_temp = pr_solutions[1][:,0:5]
        target_wave.append(target_wave_temp)
    
features_potential = np.array(features_potential)
target_energies = np.array(target_energies)
target_wave = np.array(target_wave)
target_wave_reshape = target_wave.reshape(n_sample,1024*5)

#%%

################################
### Saving Features and Targets
################################

file_potential = '/Users/maximsecor/Desktop/ANN_1DTISE/TRAINING_DATA/1d_potential.csv'
file_energies = '/Users/maximsecor/Desktop/ANN_1DTISE/TRAINING_DATA/1d_energies.csv'
file_wave = '/Users/maximsecor/Desktop/ANN_1DTISE/TRAINING_DATA/1d_wave.csv'

os.system('touch ' + file_potential)
os.system('touch ' + file_energies)
os.system('touch ' + file_wave)

df_features_potential = pd.DataFrame(features_potential)
df_target_energies = pd.DataFrame(target_energies)
df_target_wave = pd.DataFrame(target_wave_reshape)

df_features_potential.to_csv(file_potential, index = False, header=True)
df_target_energies.to_csv(file_energies, index = False, header=True)
df_target_wave.to_csv(file_wave, index = False, header=True)


