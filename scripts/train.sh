#!/bin/bash

run_id=0
n_gpus=1
gpu_offset=0
n_runs_per_gpu=2
config_file="default.yaml"


## Put your wandb info here
wandb_entity=''
wandb_project=''

wandb_mode='online'
wandb_mode='disabled'



# Data
n_states=2
n_ignores_train=''
n_ignores_val="'1;50;127'"

# fdata_train='representative_data_equal_intvls_N100000_20250723_005003.pkl'
fdata_train='representative_data_equal_intvls_20250722_235634.pkl'
fdata_train='data/'$fdata_train
fdata_val=''

# TODO: use a fixed set of training samples
n_ignores_train="'1;100;127'"
n_ignores_val=$n_ignores_train


n_epochs=500
n_samples_train=8192
n_samples_val=2048


lr_schedule='constant'
n_steps_to_eval=1000
n_steps_to_save=2000

for data_seed in 1;
do
for model_seed in 1;
do 
for lr in 1e-3;
do
for weight_decay in 0;
do
for dim in 256;
do
for heads in 1;
do
for depth in 1;
do
  gpu_id=$((run_id%n_gpus))
  gpu_id=$((gpu_id+gpu_offset))
  run_id=$((run_id+1))

  WANDB_MODE=$wandb_mode \
  CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    --config-name=$config_file \
    config_name=$config_name \
    model_seed=$model_seed \
    data.seed=$data_seed \
    data.n_states=$n_states \
    data.n_samples_train=$n_samples_train \
    data.n_samples_val=$n_samples_val \
    data.p_ignores_train=$p_ignores_train \
    data.p_ignores_val=$p_ignores_val \
    data.n_ignores_train=$n_ignores_train \
    data.n_ignores_val=$n_ignores_val \
    data.fdata_train=$fdata_train \
    data.fdata_val=$fdata_val \
    model.attn_layers.dim=$dim \
    model.attn_layers.depth=$depth \
    model.attn_layers.heads=$heads \
    training.epochs=$n_epochs \
    training.lr=$lr \
    training.lr_schedule=$lr_schedule \
    training.weight_decay=$weight_decay \
    training.n_steps_to_eval=$n_steps_to_eval \
    wandb.name="${data}_${run_id}" \
    wandb.entity=$wandb_entity \
    wandb.project=$wandb_project


  if [[ $((run_id % (n_runs_per_gpu*n_gpus))) -eq 0 ]]; then
    echo "Waiting at $run_id"
    wait
  fi

done
done
done
done
done
done
done
