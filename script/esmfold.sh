
CUDA_VISIBLE_DEVICES=0 python src/esmfold.py \
    --fasta_file data/deeploc/deeploc_test.fasta \
    --out_dir data/deeploc/esmfold_pdb \
    --fold_chunk_size 32