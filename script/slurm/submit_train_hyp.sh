# dataset: deeploc-1_binary deeploc-1_multi deepsol
dataset=deepsol
# pooling_method: attention1d mean light_attention
pooling_method=attention1d
# pdb_type: ef af
pdb_type=ef
# learning rate: 1e-3 1e-4 1e-5
lr=1e-4

sbatch --export=dataset=$dataset,pooling_method=$pooling_method,lr=$lr,pdb_type=$pdb_type --job-name=$dataset"_"$pooling_method"_"$pdb_type script/slurm/train_hyp.slurm



# tune hyperparameters for deeploc-1_binary
for pooling_method in attention1d
do
    for pdb_type in ef af
    do
        for lr in 1e-3 1e-4
        do
            sbatch --export=dataset=deeploc-1_binary,pooling_method=$pooling_method,lr=$lr,pdb_type=$pdb_type --job-name=deeploc-1_binary_"$pooling_method"_"$pdb_type" script/slurm/train_hyp.slurm
        done
    done
done

# tune hyperparameters for deeploc-1_multi
for pooling_method in mean
do
    for pdb_type in ef
    do
        for lr in 1e-3 5e-4 1e-4
        do
            sbatch --export=dataset=deeploc-1_multi,pooling_method=$pooling_method,lr=$lr,pdb_type=$pdb_type --job-name=deeploc-1_multi_"$pooling_method"_"$pdb_type" script/slurm/train_hyp.slurm
        done
    done
done

# tune hyperparameters for deepsol
for pooling_method in attention1d mean light_attention
do
    for lr in 1e-3 1e-4
    do
        sbatch --export=dataset=deepsol,pooling_method=$pooling_method,lr=$lr,pdb_type=ef --job-name=deepsol_"$pooling_method"_ef script/slurm/train_hyp.slurm
    done
done

for i in {4120..4179}
do
    scancel $i
done