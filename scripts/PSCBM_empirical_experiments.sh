#!/bin/bash
#This script performs the experiment with partial data on empirical PSCBM.
#It could be extended for evaluating the off-diagonals scaling, too.
#It is designed to be run on a SLURM cluster.
eval "$(conda shell.bash hook)"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"
export LDFLAGS=-L/usr/local/cuda/lib64/stub


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
covariance_scaling=1.0001 # Only necessary for global/empirical covariance
concept_learning='hard'
train_batch_size=128
reg_weight=1

save_model='True'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/
echo Submitting job
# 3 hard CBMs trained with different seeds. Each one is evaluated 3 times.
CBMs=(
    '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250401-162246_24835'
    # '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250616-151111_fe6d3'
    # '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250616-151111_f99e7'
    # '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250401-162246_24835'
    # '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250616-151111_fe6d3'
    # '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250616-151111_f99e7'
    # '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250401-162246_24835'
    # '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250616-151111_fe6d3'
    # '/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250616-151111_f99e7'
)

seeds=( 101 ) # 202 303 404 505 606 707 808 909 )

for i in "${!CBMs[@]}"
do
    echo "${CBMs[$i]} with seed ${seeds[$i]}"
done

cov='empirical_true'
# I limit data ratios to 2 extremes due to resource constraints.
for data_ratio in 1 0.05
do
for i in "${!CBMs[@]}"
do
        CBM=${CBMs[$i]}
        seed=${seeds[$i]}
        tag=${model}_${cov}_inference_${lr_scheduler}_${lr}_decay_${weight_decay}
        sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
        +data=$data experiment_name="${data}_${tag}_${i}" seed=$seed model.tag=$tag \
        model.load_weights=True model.weights_dir=$CBM \
        model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
        save_model=${save_model} experiment_dir=${save_model_dir} \
        model.cov_type=${cov} model.mask_density_train=${mask_density} \
        model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} \
        model.train_interventions=False model.pretrain_covariance=True \
        'model.additional_tags=[empirical,final,partial_data]'

done
done

