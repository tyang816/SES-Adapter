# ablation for without foldseek
sbatch --export=dataset_type=deeploc-1_binary,pooling_method=mean,num_labels=2,lr=1e-4,pdb_type=ef --job-name=deeploc-1_binary_mean_ef script/slurm/train_wo_foldseek.slurm
sbatch --export=dataset_type=deepsol,pooling_method=mean,num_labels=2,lr=1e-4,pdb_type=ef --job-name=deepsol_mean_ef script/slurm/train_wo_foldseek.slurm
sbatch --export=dataset_type=deeploc-1_multi,pooling_method=mean,num_labels=10,lr=1e-4,pdb_type=af --job-name=deeploc-1_multi_mean_af script/slurm/train_wo_foldseek.slurm

# ablation for without ss8
sbatch --export=dataset_type=deeploc-1_binary,pooling_method=mean,num_labels=2,lr=1e-4,pdb_type=ef --job-name=deeploc-1_binary_mean_ef script/slurm/train_wo_ss8.slurm
sbatch --export=dataset_type=deepsol,pooling_method=mean,num_labels=2,lr=1e-4,pdb_type=ef --job-name=deepsol_mean_ef script/slurm/train_wo_ss8.slurm
sbatch --export=dataset_type=deeploc-1_multi,pooling_method=mean,num_labels=10,lr=1e-4,pdb_type=af --job-name=deeploc-1_multi_mean_af script/slurm/train_wo_ss8.slurm
