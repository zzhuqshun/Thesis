#!/bin/bash

#SBATCH -o /home/users/z/zzhuqshun/Output/Results/myjob.%j.%N.out     # Output-File
#SBATCH -D /home/users/z/zzhuqshun/Thesis/02_Nbeats                        # Working Directory
#SBATCH -J Model_Run                                  # Job Name
#SBATCH --ntasks=1                                    # Number of tasks (processes)
#SBATCH --cpus-per-task=1                             # Number of CPU cores per task
#SBATCH --gres=gpu:tesla:2                            # Request 2 Tesla GPUs
#SBATCH --mem=500G                                    # Request 500 GB of memory

# Maximum walltime allowed:
#SBATCH --time=72:00:00                               # Expected run time

# Compute on GPU partition:
#SBATCH --partition=gpu

# Job status via email:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qianshun.zhu@campus.tu-berlin.de

# Load required software/libraries (e.g., Miniconda, CUDA)
module load cuda/10.1
module load miniconda3       

# Activate the Conda environment
source /home/users/z/zzhuqshun/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the Python script
python model.py
