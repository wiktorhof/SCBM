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

i=5
cov='amortized'
num_masks_train=20
lr=0.0001

for train_batch_size in 32 128
do
    for reg_weight in 0.1 0.01
    do
        for mask_density in 0.2
        do
            tag=${model}_${concept_learning}_${cov}
            sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
            +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
            model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
            save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
            model.cov_type=${cov} model.mask_density_train=${mask_density} model.num_masks_train=${num_masks_train} \
            model.inter_policy=\'prob_unc,random\' model.learning_rate=${lr} model.train_batch_size=${train_batch_size}
        done

        # Use default mask density, i.e. random up to 25%
        tag=${model}_${concept_learning}_${cov}
        sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
        +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
        model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
        save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
        model.cov_type=${cov} model.num_masks_train=${num_masks_train} \
        model.inter_policy=\'prob_unc,random\' model.learning_rate=${lr}

    done
done