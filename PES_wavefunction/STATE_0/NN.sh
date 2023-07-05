#!/bin/bash

#SBATCH --partition=pi_hammes_schiffer
#SBATCH -A hammes_schiffer
#SBATCH --job-name=pes_state0_1024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 144:00:00
#SBATCH --mem-per-cpu=12000

python ANN_pes_state0.py > ANN.out

