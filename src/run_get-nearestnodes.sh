#!/bin/sh -l
#SBATCH -N 1      
#SBATCH -n 1      
#SBATCH -c 20      
#SBATCH --partition=short
#SBATCH -t 5:00:00  # time requested in hour:minute:second

module load anaconda3/3.7
source activate amenities
which python

python get-nearestnodes.py