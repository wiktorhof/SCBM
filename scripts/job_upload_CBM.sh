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
model='CBM'
i=42
#concept_learning='hard'

save_model='True'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/
echo Submitting job

for concept_learning in 'hard'
do
for i in 13 27
do
tag=${model}_${concept_learning}_${data}
sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model +data=$data experiment_name="${data}_${tag}_${i}" \
seed=$i model.tag=$tag model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch save_model=${save_model} \
experiment_dir=${save_model_dir}
done;
done;

# for model in 'AR' 'CEM'
# do
# tag=${model}
# sbatch --output=${output_file} --mem=$mem --job-name=${tag} train.sh +model=$model +data=$data experiment_name="${data}_${tag}_${i}" seed=$i logging.project=PSCBM logging.mode=offline model.tag=$tag model.encoder_arch=$encoder_arch save_model=${save_model} experiment_dir=${save_model_dir}
# #echo Job submitted
# done;
# echo All jobs submitted
