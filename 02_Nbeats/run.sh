#!/bin/bash

#SBATCH -o ./Results/myjob.%j.out   
#SBATCH -e ./Results/myjob.%j.err 
#SBATCH -D .                     
#SBATCH -J NBeats                                 
#SBATCH --ntasks=1                              
#SBATCH --cpus-per-task=2                        
#SBATCH --gres=gpu:a100:1                            
#SBATCH --mem=32G                                    
#SBATCH --time=96:00:00                             
#SBATCH --partition=gpu

# Job status via email:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qianshun.zhu@campus.tu-berlin.de

# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate myenv

# Run the Python script
python model.py

