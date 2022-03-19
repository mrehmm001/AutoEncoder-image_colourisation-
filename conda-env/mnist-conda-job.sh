#!/bin/bash
#SBATCH --time 10          # time in minutes to reserve
#SBATCH --cpus-per-task 2  # number of cpu cores
#SBATCH --mem 4G           # memory pool for all cores
#SBATCH --gres gpu:1       # number of gpu cores
#SBATCH  -o mnist.log      # log output

# Initialise conda environment.
eval "$(conda shell.bash hook)"
conda activate test

# Run jobs.
srun -l python ../mnist.py
