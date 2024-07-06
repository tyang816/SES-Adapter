dataset=MetalIonBinding
num_labels=2
pdb_type=af
pooling_head=mean
plm_model=esm2_t30_150M_UR50D
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --plm_model ckpt/$plm_model \
    --dataset $dataset \
    --problem_type single_label_classification \
    --num_labels $num_labels \
    --pooling_method $pooling_head \
    --test_file data/$dataset/test.csv \
    --test_result_dir result/$plm_model/$dataset/inden_test \
    --metrics accuracy \
    --max_batch_token 10000 \
    --ckpt_root result \
    --ckpt_dir $plm_model/$dataset \
    --model_name "$pdb_type"_"$pooling_head"_5e-4.pt


dataset=MetalIonBinding
num_labels=2
pdb_type=ef
plm_model=esm2_t30_150M_UR50D
CUDA_VISIBLE_DEVICES=3 python eval.py \
    --plm_model ckpt/$plm_model \
    --dataset $dataset \
    --problem_type single_label_classification \
    --num_labels $num_labels \
    --pooling_method mean \
    --use_foldseek \
    --use_ss8 \
    --test_file data/CSV/MetalIonBinding/AlphaFold2/test.csv \
    --test_result_dir result/structural_quality_ablation/"$plm_model"_"$dataset"_ef_to_af \
    --metrics accuracy \
    --max_batch_token 10000 \
    --ckpt_root result \
    --ckpt_dir $plm_model/$dataset \
    --model_name "$pdb_type"_"$pooling_head"_5e-4.pt