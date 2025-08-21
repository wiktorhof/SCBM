#!/bin/bash
#This script trains PSCBM over several random instantiations
#Including multiple underlying pre-trained CBMs
#With the SCBM loss, using the hyperparameters
#Obtained by tuning.
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
model='PSCBM'

data_ratio=1
covariance_scaling=1.0001 # Only necessary for global/empirical covariance
concept_learning='hard'
train_batch_size=128
reg_weight=1

save_model='True'
save_model_dir=/path/to/file/
cd /path/to/file/

CBMs=(
    '/path/to/file/20250401-162246_24835'
    '/path/to/file/20250616-151111_fe6d3'
    '/path/to/file/20250616-151111_f99e7'
    '/path/to/file/20250401-162246_24835'
    '/path/to/file/20250616-151111_fe6d3'
    '/path/to/file/20250616-151111_f99e7'
    '/path/to/file/20250401-162246_24835'
    '/path/to/file/20250616-151111_fe6d3'
    '/path/to/file/20250616-151111_f99e7'
)


seeds=( 101 202 303 404 505 606 707 808 909 )

for i in "${!CBMs[@]}"
do
    echo "${CBMs[$i]} with seed ${seeds[$i]}"
done

# Train with amortized covariance
# cov='amortized'
# lr_scheduler=
# lr=
# weight_decay=

# for i in "${!CBMs[@]}"
# do
        # CBM=${CBMs[$i]}
        # seed=${seeds[$i]}
        # tag=${model}_${cov}_inference_${lr_scheduler}_${lr}_decay_${weight_decay}
        # sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
        # +data=$data experiment_name="${data}_${tag}_${i}" seed=$seed model.tag=$tag \
        # model.load_weights=True model.weights_dir=$CBM \
        # model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
        # save_model=${save_model} experiment_dir=${save_model_dir} \
        # model.cov_type=${cov} model.mask_density_train=${mask_density} \
        # model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} \
        # model.p_epochs=200 model.i_epochs=200 model.lr_scheduler=${lr_scheduler} \
        # model.train_interventions=False model.pretrain_covariance=True \
        # model.calculate_curves=False model.learning_rate=${lr} model.weight_decay=${weight_decay} \
        # 'model.additional_tags=[intervention,final]'

# done

# Train with global covariance
cov='global'
lr_scheduler='cosine'
lr=0.0001
weight_decay=4

for i in "${!CBMs[@]}"
do
        CBM=${CBMs[$i]}
        seed=${seeds[$i]}
        tag=${model}_${cov}_interventions_${lr_scheduler}_${lr}_decay_${weight_decay}
        sbatch --output=${output_file} --job-name=${tag} --mem=$mem train.sh +model=$model \
        +data=$data experiment_name="${data}_${tag}_${i}" seed=$seed model.tag=$tag \
        model.load_weights=True model.weights_dir=$CBM \
        model.concept_learning=$concept_learning model.encoder_arch=$encoder_arch \
        save_model=${save_model} experiment_dir=${save_model_dir}\
        model.cov_type=${cov} model.mask_density_train=${mask_density} \
        model.train_batch_size=${train_batch_size} model.reg_weight=${reg_weight} \
        model.p_epochs=100 model.i_epochs=200 model.lr_scheduler=${lr_scheduler} \
        model.train_interventions=False model.pretrain_covariance=True \
        model.calculate_curves=False model.learning_rate=${lr} model.weight_decay=${weight_decay} \
        'model.additional_tags=[intervention,final]' \

done

