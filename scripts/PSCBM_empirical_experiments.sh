#!/bin/bash
#This script performs the experiment with partial data on empirical PSCBM.
#It could be extended for evaluating the off-diagonals scaling, too.
#It is designed to be run on a SLURM cluster.
eval "$(conda shell.bash hook)"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export LIBRARY_PATH="/path/to/file/stubs:$LIBRARY_PATH"
export LDFLAGS=-L/path/to/file/stub


conda activate scbm
today=$(date +%y-%m-%d)
export WANDB_API_KEY=local-be0546acbdda04d2949d57a384bcb9552c9aede7
output_dir=/path/to/file/$today
output_file=${output_dir}/job-%J.txt
if [ ! -d ${output_dir} ]; then echo Creating log directory for $today.; mkdir ${output_dir};
else echo Log directory exists already.;
fi

data='CUB'
mem='20G'
encoder_arch='resnet18'
model='PSCBM'

data_ratio=1
covariance_scaling=1.001 # Only necessary for global/empirical covariance
concept_learning='hard'
train_batch_size=128
reg_weight=1

save_model='True'
save_model_dir=/path/to/file/
cd /path/to/file/
echo Submitting job
# 3 hard CBMs trained with different seeds. Each one is evaluated 3 times.
CBMs=(
 #   '/path/to/file/20250401-162246_24835'
    '/path/to/file/20250616-151111_fe6d3'
 # '/path/to/file/20250616-151111_f99e7'
   '/path/to/file/20250401-162246_24835'
  #  '/path/to/file/20250616-151111_fe6d3'
    '/path/to/file/20250616-151111_f99e7'
    '/path/to/file/20250401-162246_24835'
    '/path/to/file/20250616-151111_fe6d3'
    '/path/to/file/20250616-151111_f99e7'
)

seeds=(202 404 606 707 808 909)
for i in "${!CBMs[@]}"
do
    echo "${CBMs[$i]} with seed ${seeds[$i]}"
done

cov='empirical_true'
# I limit data ratios to 2 extremes due to resource constraints.
for data_ratio in 1 0.3 0.05
do
for i in "${!CBMs[@]}"
do
        CBM=${CBMs[$i]}
        seed=${seeds[$i]}
        tag=${model}_${cov}_${data_ratio}
        sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
        +data=$data experiment_name="${data}_${tag}_${seeds[$i]}" seed=$seed model.tag=$tag \
        model.load_weights=True model.weights_dir=$CBM model.data_ratio=${data_ratio} \
        model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
        save_model=${save_model} experiment_dir=${save_model_dir} \
        model.cov_type=${cov} model.mask_density_train=${mask_density} \
	model.covariance_scaling=${covariance_scaling} model.val_batch_size=64 \
        'model.additional_tags=[empirical,final,partial_data]' \
        model.calculate_interventions=False model.test=True

done
done

