#!/bin/bash

#SBATCH --partition=pi_hammes_schiffer
#SBATCH -A hammes_schiffer
#SBATCH --job-name=pes_state4_1024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 144:00:00
#SBATCH --mem-per-cpu=18000

python ANN_pes_state4.py > ANN.out

