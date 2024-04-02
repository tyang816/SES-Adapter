# dataset (esmfold & alphafold): DeepLocBinary DeepLocMulti MetalIonBinding
# dataset (esmfold only): DeepSol DeepSoluE
# plm_model (Facebook): esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D
# plm_model (RostLab): prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large
dataset=Thermostability
pdb_type=af
pooling_head=mean
plm_model=ankh-large
lr=5e-4
CUDA_VISIBLE_DEVICES=0 python train.py \
    --plm_model ckpt/$plm_model \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --pooling_dropout 0.1 \
    --dataset $dataset \
    --pdb_type $pdb_type \
    --lr $lr \
    --num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_epochs 15 \
    --max_batch_token 50000 \
    --patience 5 \
    --use_foldseek \
    --use_ss8 \
    --ckpt_root result \
    --ckpt_dir adapter_debug/$plm_model/$dataset \
    --model_name "$pdb_type"_"$pooling_head"_"$plm_model"_"$lr".pt \
    --wandb \
    --wandb_project adapter_debug \
    --wandb_run_name "$dataset"_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr"