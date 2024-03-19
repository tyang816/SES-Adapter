# dataset: deeploc-1_binary deeploc-1_multi deepsol
dataset_type=deeploc-1_multi
num_labels=10
pdb_type=ef
pooling_head=mean
plm_model=esm2_t33_650M_UR50D
CUDA_VISIBLE_DEVICES=1 python eval.py \
    --plm_model ckpt/$plm_model \
    --num_labels $num_labels \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --test_file dataset/$dataset_type/$pdb_type"_test.json" \
    --max_batch_token 100000 \
    --use_foldseek \
    --use_ss8 \
    --ckpt_dir val_acc/$dataset_type \
    --model_name "$dataset_type"_"$pdb_type"_"$pooling_head"_5e-4.pt