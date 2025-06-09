#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

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

