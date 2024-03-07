# example
sbatch --export=dataset_type=deeploc-1_multi,pooling_method=attention1d,num_labels=11 --job-name=deeploc-1_binary_mean_ef script/slurm/train_wofsss.slurm

# ablation for without foldseek
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2 --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_wofs.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2 --job-name=deepsol_"$pooling_method"_ef script/slurm/train_wofs.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10 --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train_wofs.slurm
done

# ablation for without ss8
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2 --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_woss.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2 --job-name=deepsol_"$pooling_method"_ef script/slurm/train_wofs.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10 --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train_woss.slurm
done

# ablation for without foldseek and ss8
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2 --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_wofsss.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2 --job-name=deepsol_"$pooling_method"_ef script/slurm/train_wofsss.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10 --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train_wofsss.slurm
done

# ablation for use foldseek and ss8
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2 --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train.slurm
    sbatch --export=dataset_type=deepsol,pooling_method=$pooling_method,num_labels=2 --job-name=deepsol_"$pooling_method"_ef script/slurm/train.slurm
    sbatch --export=dataset_type=deeploc-1_multi,pooling_method=$pooling_method,num_labels=10 --job-name=deeploc-1_multi_"$pooling_method"_af script/slurm/train.slurm
done