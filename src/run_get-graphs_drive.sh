#!/bin/sh -l
#SBATCH -N 2      
#SBATCH -n 2      
#SBATCH -c 30      
#SBATCH --partition=short
#SBATCH -t 20:00:00  # time requested in hour:minute:second

source activate amenities
which python

python get-graphs.py drive
