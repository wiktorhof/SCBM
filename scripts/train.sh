#!/bin/bash

now=$(date +%y:%m:%d:%H:%M)
output_filename=~/Desktop/scbm/job_outputs/$now.txt

#Lines that start with SBATCH are merely comments for bash. I cannot use variables there
#SBATCH --job-name=CBM
#SBATCH --output="/cluster/home/wiktorh/Desktop/scbm/slurm_outputs/job-%J"
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda deactivate
conda activate scbm
cd ~/Desktop/scbm/SCBM

python -u train.py "$@"