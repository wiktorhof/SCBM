#!/bin/bash
# This script is used to train baselines, namely CBM and SCBM.
eval "$(conda shell.bash hook)"
conda activate scbm
today=$(date +%y-%m-%d)
export WANDB_API_KEY=local-be0546acbdda04d2949d57a384bcb9552c9aede7
output_dir=/cluster/home/wiktorh/Desktop/scbm/slurm_outputs/$today
output_file=${output_dir}/job-%J.txt
# # if [ ! -d ${output_dir} ]; then echo Creating log directory for $today.; mkdir ${output_dir};
# # else echo Log directory exists already.;
# fi

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
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cov=global
pretrained_locations_global=(
    '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/scbm/hard/CUB/20250721-134904_539ac'
    '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/scbm/hard/CUB/20250721-214700_107ed'
    '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/scbm/hard/CUB/20250721-134904_17665'

)

for i in "${!pretrained_locations_global[@]}"
do
location=${pretrained_locations_global[$i]}
echo ${location}
save_model='False'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/
echo Submitting job
# 48 jobs in total. Each one takes some 20 minutes on 1 GPU. So totally it is 16 GPU hours.

                    tag=${model}_${cov}_${lr_scheduler}_decay_${weight_decay}_final
                    sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
                    model.load_weights=True model.weights_dir=${location} \
                    +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
                    model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
                    save_model=${save_model} experiment_dir=${save_model_dir} \
                    model.cov_type=${cov} \
                    model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} model.lr_scheduler=${lr_scheduler} \
                    'model.additional_tags=[final]'
done