# dataset: deeploc-1_binary deeploc-1_multi deepsol
dataset_type=deeploc-1_binary
num_labels=2
pdb_type=ef
pooling_head=attention1d
CUDA_VISIBLE_DEVICES=1 python train.py \
    --plm_model ckpt/esm2_t33_650M_UR50D \
    --num_labels $num_labels \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --pooling_dropout 0.25 \
    --train_file dataset/$dataset_type/$pdb_type"_train.json" \
    --val_file dataset/$dataset_type/$pdb_type"_val.json" \
    --test_file dataset/$dataset_type/$pdb_type"_test.json" \
    --lr 1e-3 \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_epochs 50 \
    --max_batch_token 100000 \
    --patience 10 \
    --monitor val_loss \
    --use_foldseek True \
    --use_ss8 True \
    --ckpt_root ckpt \
    --ckpt_dir $dataset_type \
    --model_name "$dataset_type"_"$pdb_type"_"$pooling_head"_debug.pt \
    --wandb \
    --wandb_project LocSeek_debug \
    --wandb_run_name "$dataset_type"_"$pdb_type"_"$pooling_head"