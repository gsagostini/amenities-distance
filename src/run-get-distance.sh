#!/bin/sh -l
#SBATCH -N 1      
#SBATCH -n 1      
#SBATCH -c 60      
#SBATCH --partition=short
#SBATCH -t 5:00:00

module load anaconda3/3.7
source activate amenities
which python

python get-distance.py USA81
