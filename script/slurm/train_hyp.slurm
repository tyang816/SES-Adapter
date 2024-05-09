#!/bin/bash
#SBATCH -J locseek
#SBATCH -p NvidiaA800
#SBATCH --nodelist=ZSGPU18
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16
#SBATCH --time=180-00:00:00
#SBATCH --output=/public/home/tanyang/workspace/LocSeek/log/%j.out
#SBATCH --error=/public/home/tanyang/workspace/LocSeek/log/%j.err

source ~/.bashrc
conda activate protssn
cd /public/home/tanyang/workspace/LocSeek
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export OMP_PROC_BIND=false


CUDA_VISIBLE_DEVICES=0 python train.py \
    --plm_model ckpt/esm2_t33_650M_UR50D \
    --num_attention_heads 8 \
    --pooling_method $pooling_method \
    --pooling_dropout 0.25 \
    --dataset_config dataset/$dataset/"$dataset"_"$pdb_type".json \
    --lr $lr \
    --num_workers 4 \
    --batch_size 64 \
    --max_train_epochs 50 \
    --max_seq_len 2048 \
    --patience 5 \
    --ckpt_root result \
    --ckpt_dir $dataset \
    --model_name "$dataset"_"$pdb_type"_"$pooling_method"_"$lr".pt \
    --wandb \
    --wandb_project adapter_debug \
    --wandb_run_name "$dataset"_"$pdb_type"_"$pooling_method"_"$lr"