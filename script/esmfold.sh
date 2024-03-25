
CUDA_VISIBLE_DEVICES=0 python src/esmfold.py \
    --fasta_file data/DeepSoluE/raw/training.fasta \
    --out_dir data/DeepSoluE/esmfold_pdb \
    --fold_chunk_size 64