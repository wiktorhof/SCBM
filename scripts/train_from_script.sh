#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate scbm
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
i=11

data_ratio=1
covariance_scaling=1.0001
concept_learning='hard'
cov='amortized'
train_batch_size=128
reg_weight=1
mask_density=0.2

save_model='True'
save_model_dir=/cluster/work/vogtlab/Group/wiktorh/PSCBM/models/
cd /cluster/home/wiktorh/Desktop/scbm/scripts/

for lr_scheduler in 'step' 'cosine'
do
#for lr in 0.001 0.0001 0.00001
for lr in 0.0001
do
#for weight_decay in 0 0.01
for weight_decay in 0.01
do
        #echo Running a job with inference
	#   tag=${model}_${cov}_inference_${lr_scheduler}_${lr}_decay_${weight_decay}
        #  bash train.sh +model=$model \
        #    +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
        #    model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
        #    save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
        #    model.cov_type=${cov} model.mask_density_train=${mask_density} \
        #    model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} \
        #    model.p_epochs=200 model.i_epochs=200 model.lr_scheduler=${lr_scheduler} \
	#        model.train_interventions=False model.pretrain_covariance=True \
	#        model.calculate_curves=False model.learning_rate=${lr} model.weight_decay=${weight_decay}

            
        echo Running a job with interventions
	    tag=${model}_${cov}_intervention_${lr_scheduler}_${lr}_decay_${weight_decay}_HP
            bash train.sh +model=$model \
            +data=$data experiment_name="${data}_${tag}_${i}" seed=$i model.tag=$tag \
            model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
            save_model=${save_model} experiment_dir=${save_model_dir} model.load_weights=True \
            model.cov_type=${cov} model.mask_density_train=${mask_density} \
            model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} \
            model.p_epochs=200 model.i_epochs=400 model.lr_scheduler=${lr_scheduler} \
	        model.train_interventions=True model.pretrain_covariance=False \
	        model.calculate_curves=False model.learning_rate=${lr} model.weight_decay=${weight_decay}
done
done
done
