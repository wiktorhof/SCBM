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

data_ratio=1
covariance_scaling=1.0001
concept_learning='hard'

save_model='True'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/
echo Submitting job


# for i in 1
# do
# tag=${model}_${concept_learning}_${cov}
# sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
# +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
# model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
# save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
# model.cov_type=${cov} \
# model.training_mode='joint'
# done


# cov='amortized'

# for i in 1
# do
# tag=${model}_${concept_learning}_${cov}
# sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
# +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
# model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
# save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
# model.cov_type=${cov} \
# model.inter_policy=\'prob_unc,random\'
# done

i=7
cov='amortized'
num_masks_train=20
lr=0.0001
train_batch_size=128
reg_weight=1 # Use normal
mask_density=0.2

for cov in 'amortized' 'global'
do
    # Perform covariance pretraining only and pretraining followed by interventions training
    for train_interventions in 'False'
    do
        tag=${model}_${concept_learning}_${cov}_pretrain_${train_interventions}
        sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
        +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
        model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
        save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
        model.cov_type=${cov} model.mask_density_train=${mask_density} model.num_masks_train=${num_masks_train} \
        model.inter_policy=\'prob_unc,random\' model.learning_rate=${lr} model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} \
        model.train_interventions=${train_interventions} model.curves_every=50
    done
done

cov='amortized'
reg_weight=0.01 # Use weak regularization
reg_clamp=1
for train_batch_size in 128 256 512
do
    for weight_decay in 0 0.01
    do
        # Try training amortized covariance with only interventions when gradient clamping is on. Without pretraining
        tag=${model}_${concept_learning}_${cov}_no_pretrain_clamp_decay_${weight_decay}_batch_${train_batch_size}
        sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
        +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
        model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
        save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
        model.cov_type=${cov} model.mask_density_train=${mask_density} model.num_masks_train=${num_masks_train} \
        model.inter_policy=\'prob_unc,random\' model.learning_rate=${lr} model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} \
        model.train_interventions=True model.pretrain_covariance=False model.reg_clamp=${reg_clamp} model.weight_decay=${weight_decay} model.curves_every=50
    done
done