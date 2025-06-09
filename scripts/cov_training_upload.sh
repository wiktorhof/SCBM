#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate scbm
today=$(date +%y-%m-%d)

export WANDB_API_KEY=local-be0546acbdda04d2949d57a384bcb9552c9aede7
output_dir=/cluster/home/wiktorh/Desktop/scbm/slurm_outputs/$today
output_file=${output_dir}/job-%J.txt

data='CUB'
mem='20G'
encoder_arch='resnet18'
model='PSCBM'
i=9431
concept_learning='hard'
cov='global'

save_model='True'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/
echo Submitting job

data_ratio=1
covariance_scaling=1.0001

tag=${model}_${concept_learning}_${cov}
sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
+data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
model.cov_type=${cov} model.data_ratio=${data_ratio} model.covariance_scaling=${covariance_scaling} \
model.inter_policy=\'prob_unc,random\' model.inter_strategy=\'hard,simple_perc,emp_perc,conf_interval_optimal\' \
model.training_mode='joint'