pdb_type=ef
# dataset: deeploc-1_binary deeploc-1_multi deepsol
dataset_type=deeploc-1_multi
# dataset_type=deepsol
CUDA_VISIBLE_DEVICES=0 python train.py \
    --hidden_size 1280 \
    --num_attention_heads 8 \
    --esm_model_name facebook/esm2_t33_650M_UR50D \
    --num_labels 11 \
    --pooling_method mean \
    --train_file dataset/$dataset_type/$pdb_type"_train.json" \
    --val_file dataset/$dataset_type/$pdb_type"_val.json" \
    --test_file dataset/$dataset_type/$pdb_type"_test.json" \
    --lr 1e-4 \
    --num_workers 4 \
    --batch_size 8 \
    --max_train_epochs 50 \
    --max_seq_len 2048 \
    --patience 10 \
    --ckpt_root ckpt \
    --ckpt_dir $dataset_type \
    --model_name "$dataset_type"_"$pdb_type"_debug.pt \
    --wandb_project LocSeek_debug \
    --wandb_run_name "$dataset_type"_$pdb_type