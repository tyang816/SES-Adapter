
# dataset: deeploc-1_binary deeploc-1_multi deepsol
dataset_type=deepsol
# pooling_method: attention1d mean light_attention
pooling_method=attention1d
# num_labels: 2 11
num_labels=11
# pdb_type: ef af
pdb_type=ef
# learning rate: 1e-3 1e-4 1e-5
lr=1e-4

sbatch --export=dataset_type=$dataset_type,pooling_method=$pooling_method,num_labels=$num_labels,lr=$lr,pdb_type=$pdb_type --job-name=$dataset_type"_"$pooling_method"_"$pdb_type script/slurm/train.slurm



# tune hyperparameters for deeploc-1_binary
for pooling_method in attention1d mean light_attention
do
    for pdb_type in ef af
    do
        for lr in 1e-3 1e-4 1e-5
        do
            sbatch --export=dataset_type=deeploc-1_binary,pooling_method=$pooling_method,num_labels=2,lr=$lr,pdb_type=$pdb_type --job-name=$dataset_type"_"$pooling_method"_"$pdb_type script/slurm/train.slurm
        done
    done
done