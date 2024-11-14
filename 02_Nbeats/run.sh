#!/bin/bash

#SBATCH -o /home/users/z/zzhuqshun/Output/Results/myjob.%j.%N.out     # Output-File
#SBATCH -D .                        # Working Directory
#SBATCH -J Model_Run                                  # Job Name
#SBATCH --ntasks=1                                    # Number of tasks (processes)
#SBATCH --cpus-per-task=1                             # Number of CPU cores per task
#SBATCH --gres=gpu:tesla:2                            # Request 2 Tesla GPUs
#SBATCH --mem=50G                                    # Request 500 GB of memory

# Maximum walltime allowed:
#SBATCH --time=36:00:00                               # Expected run time

# Compute on GPU partition:
#SBATCH --partition=gpu

# Job status via email:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qianshun.zhu@campus.tu-berlin.de

# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate myenv

# Run the Python script
python model.py
