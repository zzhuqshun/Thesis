#!/bin/bash

#SBATCH -o ./Results/myjob.%j.out
#SBATCH -e ./Results/myjob.%j.err
#SBATCH -D .
#SBATCH -J LSTM
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --partition=gpu

# Job status via email:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qianshun.zhu@campus.tu-berlin.de

# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate myenv

# # Create a run folder with the SLURM job ID
# RUN_FOLDER="models/run_${SLURM_JOB_ID}"
# mkdir -p "$RUN_FOLDER"

# Run the Python script, passing the folder path as an argument
python hpt.py 
