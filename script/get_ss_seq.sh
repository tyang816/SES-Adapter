data_name=deeploc

python src/data/get_ss_seq.py \
    --pdb_dir data/DeepLoc/cls10/esmfold_pdb\
    --num_workers 6 \
    --out_file data/DeepLoc/cls10/esmfold_ss.json