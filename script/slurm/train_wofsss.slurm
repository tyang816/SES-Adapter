#!/bin/bash
#SBATCH -J SES-Adapter
#SBATCH -p NvidiaA800
#SBATCH --nodelist=ZSGPU18
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=180-00:00:00
#SBATCH --output=/public/home/tanyang/workspace/SES-Adapter/log/%j.out
#SBATCH --error=/public/home/tanyang/workspace/SES-Adapter/log/%j.err

source ~/.bashrc
conda activate protssn
cd /public/home/tanyang/workspace/SES-Adapter
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export OMP_PROC_BIND=false

python train.py \
    --plm_model ckpt/$plm_model \
    --num_attention_heads 8 \
    --pooling_method $pooling_method \
    --pooling_dropout 0.1 \
    --dataset_config dataset/$dataset/"$dataset"_"$pdb_type".json \
    --lr 5e-4 \
    --num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_epochs 50 \
    --max_batch_token 120000 \
    --patience 5 \
    --ckpt_root result \
    --ckpt_dir $plm_model/$dataset \
    --model_name wofsss_"$pdb_type"_"$pooling_method"_5e-4.pt \
    --wandb \
    --wandb_entity ty_ang \
    --wandb_project adapter \
    --wandb_run_name wofsss_"$plm_model"_"$pdb_type"_"$pooling_method"_5e-4