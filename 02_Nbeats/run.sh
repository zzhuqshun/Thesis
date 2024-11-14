#!/bin/bash

#SBATCH -o ./Results/myjob.%j.out   
#SBATCH -e ./Results/myjob.%j.err 
#SBATCH -D .                     
#SBATCH -J Model_Run                                  
#SBATCH --ntasks=1                              
#SBATCH --cpus-per-task=1                        
#SBATCH --gres=gpu:a100:1                            
#SBATCH --mem=100G                                    

# Maximum walltime allowed:
#SBATCH --time=24:00:00                             

# Compute on GPU partition:
#SBATCH --partition=gpu_short

# Job status via email:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qianshun.zhu@campus.tu-berlin.de

# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate myenv

# Run the Python script
python model.py
