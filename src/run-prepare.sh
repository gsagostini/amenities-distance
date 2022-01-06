#!/bin/sh -l
#SBATCH -N 1      
#SBATCH -n 1      
#SBATCH -c 1
#SBATCH -t 5:00:00      
#SBATCH --partition=short
#SBATCH --array=1-162

fua_code=$( awk "NR==$SLURM_ARRAY_TASK_ID" ../data/d02_processed-safegraph/safegraph_fua.txt)

module load anaconda3/3.7
source activate amenities
which python

python get-preparedOD.py $fua_code