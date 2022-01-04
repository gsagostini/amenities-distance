#!/bin/sh -l
#SBATCH -N 2      
#SBATCH -n 2      
#SBATCH -c 15      
#SBATCH --partition=short
#SBATCH -t 5:00:00  # time requested in hour:minute:second

module load anaconda3/3.7
source activate amenities
which python

python get-graphs.py drive
