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

sbatch --output=${output_file} --mem=$mem train.sh +model=$model +data=$data experiment_name="${data}_${model}_${i}" seed=$i logging.project=PSCBM logging.mode=offline model.tag=$tag model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch model.j_epochs=150 model.c_epochs=100 model.t_epochs=50 save_model=${save_model} experiment_dir=${save_model_dir}