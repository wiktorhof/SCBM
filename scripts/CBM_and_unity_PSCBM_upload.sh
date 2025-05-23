#!/bin/bash
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
model='PSCBM'
i=42
#concept_learning='hard'

save_model='False'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/
echo Submitting job

tag=${model}_${concept_learning}_${data}_identity
sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True model.cov_type=indentity model.training_mode=joint ++model.CBM_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250402-115252_3c2ab/model.pth
echo Job submitted

model='CBM'
tag=${model}_${concept_learning}_${data}_baseline
sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True model.training_mode=joint ++model.weights_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250402-115252_3c2ab/model.pth
echo All jobs submitted
