#!/bin/bash

#SBATCH --partition=pi_hammes_schiffer
#SBATCH -A hammes_schiffer
#SBATCH --job-name=spectra_pes_1024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time 144:00:00
#SBATCH --mem-per-cpu=6000

python ANN_spectra_pes.py > ANN.out

