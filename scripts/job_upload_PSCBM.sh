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
i=9431
cov='empirical_predicted'
#concept_learning='hard'

save_model='False'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/
echo Submitting job


for data_ratio in 1
do
concept_learning='hard'
for covariance_scaling in 16 8 4 2 1.1 1.01 1.001 1.0001
do
tag=${model}_${concept_learning}_${cov}_r${data_ratio}_s${covariance_scaling}
sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
+data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
model.cov_type=${cov} model.data_ratio=${data_ratio} model.covariance_scaling=${covariance_scaling} \
model.inter_policy=\'prob_unc,random\' model.inter_strategy=\'hard,simple_perc,emp_perc,conf_interval_optimal\' \
model.training_mode='joint'
done
done
# concept_learning='soft'
# for data_ratio in 1
# do
# for covariance_scaling in 1 1.5 2 4 8
# do
# tag=${model}_${concept_learning}_r${data_ratio}_s${covariance_scaling}
# sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
# +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
# model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
# save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
# model.cov_type=empirical_true model.data_ratio=${data_ratio} model.covariance_scaling=${covariance_scaling} \
# model.inter_policy=\'random,prob_unc\' model.inter_strategy=\'simple_perc,emp_perc,conf_interval_optimal\' \
# model.training_mode='joint'
# done
# done
#for model in 'AR' 'CEM'
#do
#tag=${model}_${data}
#sbatch --output=${output_file} --mem=$mem --job-name=${tag} train.sh +model=$model +data=$data \
#experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
#model.encoder_arch=$encoder_arch save_model=True experiment_dir=${save_model_dir}
# echo Job submitted

#done;
echo All jobs submitted
