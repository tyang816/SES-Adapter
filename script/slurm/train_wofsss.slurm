#!/bin/bash
#SBATCH -J locseek
#SBATCH -p NvidiaA800
#SBATCH --nodelist=ZSGPU18
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=180-00:00:00
#SBATCH --output=/public/home/tanyang/workspace/LocSeek/log/%j.out
#SBATCH --error=/public/home/tanyang/workspace/LocSeek/log/%j.err


source ~/.bashrc
conda activate protssn
cd /public/home/tanyang/clash-for-linux-backup-main
bash start.sh
proxy_on
cd /public/home/tanyang/workspace/LocSeek
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export OMP_PROC_BIND=false

python train.py \
    --plm_model ckpt/$plm_model \
    --num_labels $num_labels \
    --num_attention_heads 8 \
    --pooling_method $pooling_method \
    --pooling_dropout 0.1 \
    --train_file dataset/$dataset_type/af"_train.json" \
    --val_file dataset/$dataset_type/af"_val.json" \
    --test_file dataset/$dataset_type/af"_test.json" \
    --lr 5e-4 \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_epochs 10 \
    --max_batch_token 100000 \
    --patience 3 \
    --monitor val_acc \
    --ckpt_root ckpt \
    --ckpt_dir val_acc/$dataset_type \
    --model_name wofsssr_"$dataset_type"_"af"_"$pooling_method"_5e-4.pt \
    --wandb \
    --wandb_project LocSeek \
    --wandb_run_name wofsssr_"$dataset_type"_"af"_"$pooling_method"_5e-4