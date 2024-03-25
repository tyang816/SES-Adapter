#!/bin/bash -l
#SBATCH -J locseek
#SBATCH -p NvidiaA800
#SBATCH --nodelist=ZSGPU18
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -N 1
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
    --plm_model ckpt/esm2_t33_650M_UR50D \
    --num_labels $num_labels \
    --num_attention_heads $num_heads \
    --pooling_method mean \
    --pooling_dropout $pooling_dropout \
    --train_file dataset/$dataset_type/ef"_train.json" \
    --val_file dataset/$dataset_type/ef"_val.json" \
    --test_file dataset/$dataset_type/ef"_test.json" \
    --lr 5e-4 \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_epochs 10 \
    --max_batch_token 50000 \
    --patience 3 \
    --use_foldseek \
    --use_ss8 \
    --monitor val_acc \
    --ckpt_root ckpt \
    --ckpt_dir val_acc/$dataset_type \
    --model_name "$dataset_type"_"ef"_5e-4_"$num_heads"_"$pooling_dropout".pt \
    --wandb \
    --wandb_project LocSeek \
    --wandb_run_name "$dataset_type"_"ef"_5e-4_"$num_heads"_"$pooling_dropout"