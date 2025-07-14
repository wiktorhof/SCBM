#!/bin/bash
#SBATCH --job-name=CBM
#SBATCH --output="/cluster/home/wiktorh/Desktop/scbm/slurm_outputs/job-%J"
#SBATCH --cpus-per-task=2
#SBATCH --time=0-04:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#Lines that start with SBATCH are merely comments for bash. I cannot use variables there
#Lines that start with SBATCH are merely comments for bash. I cannot use variables there

now=$(date +%y:%m:%d:%H:%M)
output_filename=~/Desktop/scbm/job_outputs/$now.txt

cd /cluster/home/wiktorh/Desktop/scbm
echo $(which python)
#source ~/.bashrc
conda init
conda activate scbm
echo Conda environment activated
echo
echo $(which python)

python -u train.py "$@"
#python ~/Desktop/test/environment_test.py

