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
cd /public/home/tanyang/clash-for-linux-backup-main
bash start.sh
proxy_on
cd /public/home/tanyang/workspace/LocSeek


CUDA_VISIBLE_DEVICES=0 python train.py \
    --plm_model ckpt/esm2_t33_650M_UR50D \
    --num_labels $num_labels \
    --num_attention_heads 8 \
    --pooling_method $pooling_method \
    --pooling_dropout 0.25 \
    --train_file dataset/$dataset_type/ef"_train.json" \
    --val_file dataset/$dataset_type/ef"_val.json" \
    --test_file dataset/$dataset_type/ef"_test.json" \
    --lr 1e-3 \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_epochs 20 \
    --max_batch_token 100000 \
    --patience 5 \
    --use_foldseek \
    --monitor val_loss \
    --ckpt_root ckpt \
    --ckpt_dir $dataset_type \
    --model_name woss_"$dataset_type"_"ef"_"$pooling_method"_1e-3.pt \
    --wandb \
    --wandb_project LocSeek \
    --wandb_run_name "$dataset_type"_"ef"_"$pooling_method"_1e-3