
CUDA_VISIBLE_DEVICES=0 python train.py \
    --hidden_size 1280 \
    --intermediate_size 2560 \
    --num_attention_heads 8 \
    --esm_model_name facebook/esm2_t33_650M_UR50D \
    --num_labels 2 \
    --train_file data/deeploc/binary/af_train.json \
    --val_file data/deeploc/binary/af_train.json \
    --test_file data/deeploc/binary/af_train.json \
    --lr 1e-4 \
    --max_batch_token 5000 \
    --max_train_epochs 30 \
    --model_name deeploc_debug.pt \
    --ckpt_root ckpt \
    --ckpt_dir deeploc \
    --wandb_project LocSeek_debug

