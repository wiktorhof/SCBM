#!/bin/bash

today=$(date +%y-%m-%d)

output_file=/cluster/home/wiktorh/Desktop/scbm/slurm_outputs/$today/job-%J.txt

data='CUB'
mem='20G'
encoder_arch='resnet18'
model='CBM'
i=42
concept_learning='hard'
tag=${model}_${concept_learning}

save_model='True'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/

sbatch --output=${output_file} --mem=$mem train.sh +model=$model +data=$data \
experiment_name=${data}_${model}_global_${i} \
+model.CBM_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/cbm/hard/CUB/20250402-115252_3c2ab/model.pth \
seed=$i logging.project=PSCBM logging.mode=offline model.tag=$tag \
model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
model.cov_type=global save_model=${save_model} experiment_dir=${save_model_dir} \
model.reg_precision=l1 model.covariance_scaling=1.0001 model.load_weights=True \
model.inter_strategy='hard' model.inter_policy='random,prob_unc'