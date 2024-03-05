# ablation for without foldseek
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2,lr=1e-3,pdb_type=ef --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_wo_foldseek.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2,lr=1e-3,pdb_type=ef --job-name=deepsol_"$pooling_method"_ef script/slurm/train_wo_foldseek.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10,lr=1e-3,pdb_type=ef --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train_wo_foldseek.slurm
done

# ablation for without ss8
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2,lr=1e-3,pdb_type=ef --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_wo_ss8.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2,lr=1e-3,pdb_type=ef --job-name=deepsol_"$pooling_method"_ef script/slurm/train_wo_ss8.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10,lr=1e-3,pdb_type=ef --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train_wo_ss8.slurm
done