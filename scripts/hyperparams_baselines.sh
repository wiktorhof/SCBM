#!/bin/bash
# This script is used to perform hyperparameter tuning for baselines, namely
# CBM and SCBM. LR is 10^-4, tuned are only the schedule and weight decay, 
# in order to spare computation resources.
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

i=11

save_model='True'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/
echo Submitting job
# 48 jobs in total. Each one takes some 20 minutes on 1 GPU. So totally it is 16 GPU hours.
for cov in 'amortized' 'global'
do
for lr_scheduler in 'cosine'
do
for weight_decay in 0 0.01
do
                    tag=${model}_${cov}_inference_${lr_scheduler}_${lr}_decay_${weight_decay}
                    sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
                    +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
                    model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
                    save_model=${save_model} experiment_dir=${save_model_dir} \
                    model.cov_type=${cov} model.weight_decay=${weight_decay} \
                    model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} model.lr_scheduler=${lr_scheduler} \
                    'model.additional_tags=[hyperparams_SCBM]' \
                    model.calculate_interventions=False
done
done
done


model='CBM'

for lr_scheduler in 'cosine'
do
for weight_decay in 0 0.01
do
                    tag=${model}_${lr_scheduler}_decay_${weight_decay}
                    sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
                    +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
                    model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
                    save_model=${save_model} experiment_dir=${save_model_dir} model.weight_decay=${weight_decay} \
                    model.train_batch_size=${train_batch_size} model.lr_scheduler=${lr_scheduler} \
                    'model.additional_tags=[hyperparams_CBM]' \
                    model.calculate_interventions=False
done
done
