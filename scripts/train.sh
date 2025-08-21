#!/bin/bash
#SBATCH --job-name=CBM
#SBATCH --output="/path/to/file/job-%J"
#SBATCH --cpus-per-task=2
#SBATCH --time=0-24:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

now=$(date +%y:%m:%d:%H:%M)
output_filename=~/path/to/file/$now.txt

cd /path/to/file/scbm
echo $(which python)
#source ~/.bashrc
conda init
conda activate scbm
echo Conda environment activated
echo
echo $(which python)

python -u train.py "$@"

