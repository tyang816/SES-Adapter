dataset="deeploc"
CUDA_VISIBLE_DEVICES=0 python src/esmfold.py \
    --fasta_file data/$dataset/deeploc_complete_dataset.fasta \
    --out_dir data/$dataset/esmfold_pdb \
    --fold_chunk_size 64