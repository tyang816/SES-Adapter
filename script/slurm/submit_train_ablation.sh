# example
sbatch --export=dataset_type=deeploc-1_binary,pooling_method=mean,num_labels=2, --job-name=deeploc-1_binary_mean_ef script/slurm/train_ablation.slurm

# ablation for without foldseek
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2,use_foldseek=False,use_ss8=True --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_ablation.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2,use_foldseek=False,use_ss8=True --job-name=deepsol_"$pooling_method"_ef script/slurm/train_ablation.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10,use_foldseek=False,use_ss8=True --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train_ablation.slurm
done

# ablation for without ss8
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2,use_foldseek=True,use_ss8=False --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_ablation.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2,use_foldseek=True,use_ss8=False --job-name=deepsol_"$pooling_method"_ef script/slurm/train_ablation.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10,use_foldseek=True,use_ss8=False --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train_ablation.slurm
done

# ablation for without foldseek and ss8
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2,use_foldseek=False,use_ss8=False --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_ablation.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2,use_foldseek=False,use_ss8=False --job-name=deepsol_"$pooling_method"_ef script/slurm/train_ablation.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10,use_foldseek=False,use_ss8=False --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train_ablation.slurm
done

# ablation for use foldseek and ss8
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2,use_foldseek=True,use_ss8=True --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_ablation.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2,use_foldseek=True,False,use_ss8=True --job-name=deepsol_"$pooling_method"_ef script/slurm/train_ablation.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10,use_foldseek=True,use_ss8=True --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train_ablation.slurm
done