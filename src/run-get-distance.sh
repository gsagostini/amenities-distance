#!/bin/sh -l
#SBATCH -N 1      
#SBATCH -n 1      
#SBATCH -c 4
#SBATCH -t 5:30:00      
#SBATCH --partition=short
#SBATCH --array=1-161

fua_code=$( awk "NR==$SLURM_ARRAY_TASK_ID" ../data/d02_processed-safegraph/safegraph_fua.txt)

module load anaconda3/3.7
source activate amenities
which python

python get-distance.py $fua_code
