
CUDA_VISIBLE_DEVICES=0 python src/esmfold.py \
    --fasta_file data/DeepLoc/multi_loc_unfold.fasta \
    --out_dir data/DeepLoc/cls10/esmfold_pdb \
    --fold_chunk_size 32