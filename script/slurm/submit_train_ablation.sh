# example
sbatch --export=dataset=deeploc-1_multi,pooling_method=attention1d --job-name=deeploc-1_binary_mean_ef script/slurm/train_wofsss.slurm

############################################################################################################

# ablation
# dataset esmfold & alphafold: deeploc-1_binary deeploc-1_multi MetalIonBinding
# dataset esmfold only: deepsol DeepSoluE
# plm_model: esm2_t30_150M_UR50D esm2_t33_650M_UR50D prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large
for dataset in DeepSoluE
do
    for plm_model in esm2_t30_150M_UR50D esm2_t33_650M_UR50D prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd
    do
        for pooling_method in mean attention1d
        do
            sbatch --export=dataset=$dataset,pdb_type=ef,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_ef script/slurm/train_wofs.slurm
            sbatch --export=dataset=$dataset,pdb_type=ef,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_ef script/slurm/train_woss.slurm
            sbatch --export=dataset=$dataset,pdb_type=ef,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_ef script/slurm/train.slurm
            sbatch --export=dataset=$dataset,pdb_type=ef,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_ef script/slurm/train_wofsss.slurm

            # sbatch --export=dataset=$dataset,pdb_type=af,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_af script/slurm/train_wofs.slurm
            # sbatch --export=dataset=$dataset,pdb_type=af,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_af script/slurm/train_woss.slurm
            # sbatch --export=dataset=$dataset,pdb_type=af,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_af script/slurm/train.slurm
            # sbatch --export=dataset=$dataset,pdb_type=af,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_af script/slurm/train_wofsss.slurm
        done
    done
done


############################################################################################################

# ablation for deeploc-1_binary
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset=deeploc-1_binary,pooling_method=$pooling_method --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train.slurm
    sbatch --export=dataset=deeploc-1_binary,pooling_method=$pooling_method --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_wofsss.slurm
    sbatch --export=dataset=deeploc-1_binary,pooling_method=$pooling_method --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_wofs.slurm
    sbatch --export=dataset=deeploc-1_binary,pooling_method=$pooling_method --job-name=deeploc-1_binary_"$pooling_method"_ef script/slurm/train_woss.slurm
done

############################################################################################################

# ablation for head number and pooling dropout
for head_num in 2 4 8 16 32
do
    for pooling_dropout in 0 0.1 0.25 0.5
    do
        sbatch --export=dataset=deeploc-1_binary,num_heads=$head_num,pooling_dropout=$pooling_dropout --job-name=deeploc-1_binary_attention1d_"$head_num"_"$pooling_dropout"_ef script/slurm/train_head.slurm
        sbatch --export=dataset=deeploc-1_multi,num_heads=$head_num,pooling_dropout=$pooling_dropout --job-name=deeploc-1_multi_attention1d_"$head_num"_"$pooling_dropout"_ef script/slurm/train_head.slurm
        sbatch --export=dataset=deepsol,num_heads=$head_num,pooling_dropout=$pooling_dropout --job-name=deepsol_attention1d_"$head_num"_"$pooling_dropout"_ef script/slurm/train_head.slurm
    done
done