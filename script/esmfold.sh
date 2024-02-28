
CUDA_VISIBLE_DEVICES=1 python src/esmfold.py \
    --fasta_file data/deeploc/deeploc_val.fasta \
    --out_dir data/deeploc/esmfold_pdb \
    --out_info_file data/deeploc/esmfold_val.csv \
    --fold_chunk_size 256