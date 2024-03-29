data_name=EC
data_type=esmfold
python src/data/get_ss_seq.py \
    --pdb_dir data/$data_name/"$data_type"_pdb\
    --num_workers 6 \
    --out_file data/$data_name/"$data_type"_ss.json