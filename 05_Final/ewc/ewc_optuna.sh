#!/bin/bash

#SBATCH -o ./Results/myjob.%j.out
#SBATCH -e ./Results/myjob.%j.err
#SBATCH -D ../
#SBATCH -J ewc_optuna
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu

# Job status via email:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qianshun.zhu@campus.tu-berlin.de

export TZ='Europe/Berlin'   

# ——— Print nodes information ———
echo "=== Job Start: $(date) ==="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Node List:        $SLURM_NODELIST"
echo "Requested GPUs:   $SLURM_JOB_GPUS"
echo "CUDA Devices:     $CUDA_VISIBLE_DEVICES"
echo "Running on host:  $(hostname)"
echo
echo "=== GPU Details via nvidia-smi ==="
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu,temperature.gpu \
           --format=csv,noheader
echo "==================================="


# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate myenv

# Run the Python script, passing the folder path as an argument
python -m ewc.ewc_optuna


