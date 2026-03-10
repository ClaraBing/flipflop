#!/bin/bash

# SLURM parameters
#SBATCH --partition=kempner # Partition to submit to
#SBATCH --account=kempner_bingbin_lab
#SBATCH -n 16 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --job-name=train
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=300000 # Memory total in MB
#SBATCH --array=0-4%12
#SBATCH --output=logs/train/%a.log
#SBATCH --error=logs/train/%a.err

cd /n/home08/bingbin/flipflop/jax

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p logs/train

# Load required modules
module load python

export PATH=/n/home08/bingbin/miniconda3/bin:$PATH

source /n/home08/bingbin/miniconda3/etc/profile.d/conda.sh

conda deactivate
conda activate common

cuda_id=0
n_gpus=1

config_file="default.yaml"

## Put your wandb info here
wandb_entity='bingbin'
wandb_project='flipflop2025'

wandb_mode='online'
# wandb_mode='disabled'

# Data
length=64
n_states=2
# n_ignores_train='1;120'
# n_ignores_val="'1;50;80;120;127'"
n_ignores_train='1;60'
n_ignores_val='1;20;36;50;63'

n_epochs=1
# n_samples_train=
n_samples_train=1024000
n_samples_val=2048

lr_schedule='cosine'
n_steps_to_eval=1000
n_steps_to_save=2000

### Sweep parameters
data_seeds=(0)
#  1 2 3 4 5 6 7 8 9)
model_seeds=(0)
lr_low=3e-4
lr_high=3e-3
num_lrs=5
lrs=( $(python3 -c "import numpy as np; print(' '.join(f'{x:.3g}' for x in np.geomspace($lr_low, $lr_high, $num_lrs)))") )
# lrs=(0)
weight_decays=(0)
Ds=(128)  # d_model (was dims)
heads_lst=(1)  # n_heads (was heads_lst)
Ls=(1)  # n_layers (was depths)


### Loops in a single job
seed_bash_loop=(0 1 2 3 4 5)

IDX=$SLURM_ARRAY_TASK_ID
REMAIN=$IDX

# data_seed index
N2=$(( ${#model_seeds[@]} * ${#lrs[@]} * ${#weight_decays[@]} * ${#Ds[@]} * ${#heads_lst[@]} * ${#Ls[@]} ))
DATA_SEED_IDX=$(( REMAIN / N2 ))
REMAIN=$(( REMAIN % N2 ))

# model_seed index
N3=$(( ${#lrs[@]} * ${#weight_decays[@]} * ${#Ds[@]} * ${#heads_lst[@]} * ${#Ls[@]} ))
MODEL_SEED_IDX=$(( REMAIN / N3 ))
REMAIN=$(( REMAIN % N3 ))

# lr index
N4=$(( ${#weight_decays[@]} * ${#Ds[@]} * ${#heads_lst[@]} * ${#Ls[@]} ))
LR_IDX=$(( REMAIN / N4 ))
REMAIN=$(( REMAIN % N4 ))

# weight_decay index
N5=$(( ${#Ds[@]} * ${#heads_lst[@]} * ${#Ls[@]} ))
WEIGHT_DECAY_IDX=$(( REMAIN / N5 ))
REMAIN=$(( REMAIN % N5 ))

# D (d_model) index
N6=$(( ${#heads_lst[@]} * ${#Ls[@]} ))
D_IDX=$(( REMAIN / N6 ))
REMAIN=$(( REMAIN % N6 ))

# H (n_heads) index
N7=$(( ${#Ls[@]} ))
H_IDX=$(( REMAIN / N7 ))
REMAIN=$(( REMAIN % N7 ))

# L (n_layers) index
L_IDX=$REMAIN

data_seed=${data_seeds[DATA_SEED_IDX]}
# model_seed=${model_seeds[MODEL_SEED_IDX]}
model_seed=$data_seed
lr=${lrs[LR_IDX]}
weight_decay=${weight_decays[WEIGHT_DECAY_IDX]}
D=${Ds[D_IDX]}
H=${heads_lst[H_IDX]}
L=${Ls[L_IDX]}

# Calculate K (d_head) as D // H, and M (d_mlp) as 4 * D
K=$((D / H))
M=$((4 * D))

echo $SLURMD_NODENAME
echo $HOSTNAME
echo "data_seed=$data_seed, model_seed=$model_seed, lr=$lr, weight_decay=$weight_decay, D=$D, H=$H, L=$L, K=$K, M=$M"



wandb_name="len${length}"

n_runs_per_gpu=3
cnt=0

for seed in ${seed_bash_loop[@]};
do
    data_seed=$seed
    model_seed=$seed
    WANDB_MODE=$wandb_mode \
    CUDA_VISIBLE_DEVICES=$cuda_id python train.py \
    --config-name=$config_file \
    config_name=$config_name \
    model_seed=$model_seed \
    data.seed=$data_seed \
    data.length=$length \
    data.n_states=$n_states \
    data.n_samples_train=$n_samples_train \
    data.n_samples_val=$n_samples_val \
    data.p_ignores_train=$p_ignores_train \
    data.p_ignores_val=$p_ignores_val \
    data.n_ignores_train=$n_ignores_train \
    data.n_ignores_val=$n_ignores_val \
    data.fdata_train=$fdata_train \
    data.fdata_val=$fdata_val \
    model.D=$D \
    model.L=$L \
    model.M=$M \
    model.H=$H \
    model.K=$K \
    training.epochs=$n_epochs \
    training.lr=$lr \
    training.lr_schedule=$lr_schedule \
    training.weight_decay=$weight_decay \
    training.n_steps_to_eval=$n_steps_to_eval \
    wandb.name=$wandb_name \
    wandb.entity=$wandb_entity \
    wandb.project=$wandb_project &

    cnt=$((cnt+1))
    if [[ $((cnt % n_runs_per_gpu)) -eq 0 ]]; then
        echo "Waiting at $cnt"
        wait
    fi
done
wait
