#!/bin/bash
# This script is used to train baselines, namely CBM and SCBM.
eval "$(conda shell.bash hook)"
conda activate scbm
today=$(date +%y-%m-%d)

output_dir=/path/to/file/$today
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
lr=0.0001
lr_scheduler='step'
weight_decay=0.01
save_model='True'

for i in 404 505 606 # 3 random seeds
do

save_model_dir=/path/to/file/
cd /path/to/file/

for cov in 'global' 'amortized'
do
                    tag=${model}_${cov}_${lr_scheduler}_decay_${weight_decay}_final
                    sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
                    +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
                    model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
                    save_model=${save_model} experiment_dir=${save_model_dir} \
                    model.cov_type=${cov} model.weight_decay=${weight_decay} \
                    model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} model.lr_scheduler=${lr_scheduler} \
                    model.val_batch_size=256 'model.additional_tags=[final]'

done
done


model='CBM'
for i in 404 505 606 # 3 random seeds
do

save_model_dir=/path/to/file/
cd /path/to/file/

tag=${model}_${cov}_${lr_scheduler}_decay_${weight_decay}_final
sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
+data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
save_model=${save_model} experiment_dir=${save_model_dir} \
model.weight_decay=${weight_decay} \
model.train_batch_size=${train_batch_size} model.lr_scheduler=${lr_scheduler} \
model.val_batch_size=256 'model.additional_tags]'


done