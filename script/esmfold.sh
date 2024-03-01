
CUDA_VISIBLE_DEVICES=1 python src/esmfold.py \
    --fasta_file data/deeploc/deeploc.fasta \
    --out_dir data/deeploc/esmfold_pdb \
    --fold_chunk_size 32