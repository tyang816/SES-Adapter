# dataset: deeploc-1_binary deeploc-1_multi deepsol
dataset_type=deeploc-1_multi
num_labels=10
pdb_type=ef
pooling_head=mean
plm_model_name=prot_bert
CUDA_VISIBLE_DEVICES=0 python train.py \
    --plm_model ckpt/$plm_model_name \
    --num_labels $num_labels \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --pooling_dropout 0.1 \
    --train_file dataset/$dataset_type/$pdb_type"_train.json" \
    --val_file dataset/$dataset_type/$pdb_type"_val.json" \
    --test_file dataset/$dataset_type/$pdb_type"_test.json" \
    --lr 1e-3 \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_epochs 10 \
    --max_batch_token 100000 \
    --patience 3 \
    --use_foldseek \
    --use_ss8 \
    --monitor val_acc \
    --ckpt_root ckpt \
    --ckpt_dir debug/$plm_model_name/$dataset_type \
    --model_name "$dataset_type"_"$pdb_type"_"$pooling_head"_debug.pt \
    --wandb_project LocSeek_debug \
    --wandb_run_name "$dataset_type"_"$pdb_type"_"$pooling_head"