#!/bin/bash
# This script is used to train baselines, namely CBM and SCBM.
eval "$(conda shell.bash hook)"
conda activate scbm
today=$(date +%y-%m-%d)
export WANDB_API_KEY=local-be0546acbdda04d2949d57a384bcb9552c9aede7
output_dir=/cluster/home/wiktorh/Desktop/scbm/slurm_outputs/$today
output_file=${output_dir}/job-%J.txt
if [ ! -d ${output_dir} ]; then echo Creating log directory for $today.; mkdir ${output_dir};
else echo Log directory exists already.;
fi

data='CUB'
mem='20G'
encoder_arch='resnet18'
model='SCBM'

concept_learning='hard'
cov='amortized'
train_batch_size=128
reg_weight=1
lr=0.0001
lr_scheduler='step'
weight_decay=0.01

for i in 404 505 606 707 # 3 random seeds
do

save_model='True'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/
echo Submitting job
# 48 jobs in total. Each one takes some 20 minutes on 1 GPU. So totally it is 16 GPU hours.
for cov in 'global'
do
                    tag=${model}_${cov}_${lr_scheduler}_decay_${weight_decay}_final
                    sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
                    +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
                    model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
                    save_model=${save_model} experiment_dir=${save_model_dir} \
                    model.cov_type=${cov} model.weight_decay=${weight_decay} \
                    model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} model.lr_scheduler=${lr_scheduler} \
                    model.val_batch_size=256 'model.additional_tags=[final,conf_interval]'

done
done
