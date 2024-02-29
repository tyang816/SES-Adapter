data_name=deeploc

python src/data/get_ss_seq.py \
    --pdb_dir data/$data_name/esmfold_pdb \
    --num_workers 6 \
    --out_file data/$data_name/esmfold_ss.json