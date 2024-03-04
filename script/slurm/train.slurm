#!/bin/bash
#SBATCH -J locseek
#SBATCH -p NvidiaA800
#SBATCH --nodelist=ZSGPU18
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16
#SBATCH --time=180-00:00:00
#SBATCH --output=/public/home/tanyang/workspace/DATA/esmfold/log/%j.out
#SBATCH --error=/public/home/tanyang/workspace/DATA/esmfold/log/%j.err

proxy_on
cd /public/home/tanyang/workspace/LocSeek
conda activate protssn


CUDA_VISIBLE_DEVICES=0 python train.py \
    --hidden_size 1280 \
    --num_attention_heads 8 \
    --esm_model_name facebook/esm2_t33_650M_UR50D \
    --num_labels $num_labels \
    --pooling_method $pooling_method \
    --train_file dataset/$dataset_type/$pdb_type"_train.json" \
    --val_file dataset/$dataset_type/$pdb_type"_val.json" \
    --test_file dataset/$dataset_type/$pdb_type"_test.json" \
    --lr $lr \
    --num_workers 4 \
    --batch_size 64 \
    --max_train_epochs 50 \
    --max_seq_len 2048 \
    --patience 10 \
    --ckpt_root ckpt \
    --ckpt_dir $dataset_type \
    --model_name "$dataset_type"_$pdb_type_"$pooling_method"_"$lr".pt \
    --wandb \
    --wandb_project LocSeek \
    --wandb_run_name "$dataset_type"_$pdb_type_"$pooling_method"_"$lr"