# dataset: deeploc-1_binary deeploc-1_multi deepsol deepsolue
# plm_model: prot_t5_xl_uniref50 esm2_t33_650M_UR50D
dataset=MetalIonBinding
pdb_type=af
pooling_head=mean
plm_model=ankh-large
lr=5e-4
CUDA_VISIBLE_DEVICES=1 python train.py \
    --plm_model ckpt/$plm_model \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --pooling_dropout 0.1 \
    --train_file dataset/$dataset/$pdb_type"_train.json" \
    --val_file dataset/$dataset/$pdb_type"_val.json" \
    --test_file dataset/$dataset/$pdb_type"_test.json" \
    --lr $lr \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_epochs 10 \
    --max_batch_token 50000 \
    --patience 3 \
    --use_foldseek \
    --use_ss8 \
    --monitor val_acc \
    --ckpt_root ckpt \
    --ckpt_dir debug/$plm_model/$dataset \
    --model_name "$dataset"_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr"_debug.pt \
    --wandb \
    --wandb_project LocSeek_debug \
    --wandb_run_name "$dataset"_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr"