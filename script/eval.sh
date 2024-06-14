dataset=DeepLocMulti
num_labels=10
pdb_type=ESMFold
pooling_head=mean
plm_model=esm2_t33_650M_UR50D
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --plm_model ckpt/$plm_model \
    --dataset $dataset \
    --problem_type single_label_classification \
    --num_labels $num_labels \
    --pooling_method $pooling_head \
    --test_file data/$dataset/inden_test.csv \
    --test_result_dir result/$plm_model/$dataset/inden_test \
    --metrics accuracy,mcc,auc,precision,recall,f1 \
    --max_batch_token 10000 \
    --ckpt_root result \
    --ckpt_dir $plm_model/$dataset \
    --model_name "$pdb_type"_"$pooling_head"_5e-4.pt