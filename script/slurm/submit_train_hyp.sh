# dataset: DeepLocBinary DeepLocMulti DeepSol
dataset=DeepSol
# pooling_method: attention1d mean light_attention
pooling_method=attention1d
# pdb_type: ESMFold AlphaFold2
pdb_type=ESMFold
# learning rate: 1e-3 1e-4 1e-5
lr=1e-4

sbatch --export=dataset=$dataset,pooling_method=$pooling_method,lr=$lr,pdb_type=$pdb_type --job-name=$dataset"_"$pooling_method"_"$pdb_type script/slurm/train_hyp.slurm



# tune hyperparameters for DeepLocBinary
for pooling_method in attention1d
do
    for pdb_type in ESMFold AlphaFold2
    do
        for lr in 1e-3 1e-4
        do
            sbatch --export=dataset=DeepLocBinary,pooling_method=$pooling_method,lr=$lr,pdb_type=$pdb_type --job-name=deeploc-1_binary_"$pooling_method"_"$pdb_type" script/slurm/train_hyp.slurm
        done
    done
done

# tune hyperparameters for DeepLocMulti
for pooling_method in mean
do
    for pdb_type in ESMFold
    do
        for lr in 1e-3 5e-4 1e-4
        do
            sbatch --export=dataset=DeepLocMulti,pooling_method=$pooling_method,lr=$lr,pdb_type=$pdb_type --job-name=deeploc-1_multi_"$pooling_method"_"$pdb_type" script/slurm/train_hyp.slurm
        done
    done
done

# tune hyperparameters for DeepSol
for pooling_method in attention1d mean light_attention
do
    for lr in 1e-3 1e-4
    do
        sbatch --export=dataset=DeepSol,pooling_method=$pooling_method,lr=$lr,pdb_type=ESMFold --job-name=deepsol_"$pooling_method"_ESMFold script/slurm/train_hyp.slurm
    done
done

for i in {6231..6461}
do
    scancel $i
done