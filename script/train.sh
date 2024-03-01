pdb_type=ef
dataset_type=deeploc-1_binary
# data/deeploc/binary/raw/train.json
# data/deeploc/binary/$data_type"_train.json"
CUDA_VISIBLE_DEVICES=1 python train.py \
    --hidden_size 1280 \
    --num_attention_heads 8 \
    --esm_model_name facebook/esm2_t33_650M_UR50D \
    --num_labels 2 \
    --pooling_method attention1d \
    --train_file dataset/$dataset_type/$pdb_type"_train.json" \
    --val_file dataset/$dataset_type/$pdb_type"_val.json" \
    --test_file dataset/$dataset_type/$pdb_type"_test.json" \
    --lr 1e-3 \
    --num_workers 4 \
    --batch_size 64 \
    --max_train_epochs 5 \
    --max_seq_len 2048 \
    --ckpt_root ckpt \
    --ckpt_dir $dataset_type \
    --model_name "$dataset_type"_"$pdb_type"_debug.pt \
    --wandb \
    --wandb_project LocSeek_debug \
    --wandb_run_name "$dataset_type"_$pdb_type