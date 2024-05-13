# example
sbatch --export=dataset=DeepLocMulti,pooling_method=attention1d --job-name=deeploc-1_binary_mean_ESMFold script/slurm/train_wofsss.slurm

############################################################################################################

# ablation
# dataset (esmfold & alphAlphaFold2old): DeepLocBinary DeepLocMulti MetalIonBinding Thermostability EC MF BP CC
# dataset (esmfold only): DeepSol DeepSoluE
# plm_model (Facebook): esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D
# plm_model (RostLab): prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large
for dataset in DeepLocBinary DeepLocMulti
do
    for plm_model in prot_bert prot_bert_bfd ankh-base ankh-large
    do
        for pooling_method in mean
        do
            sbatch --export=dataset=$dataset,pdb_type=AlphaFold2,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_AlphaFold2 script/slurm/train.slurm
            sbatch --export=dataset=$dataset,pdb_type=AlphaFold2,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_AlphaFold2 script/slurm/train_wofs.slurm
            sbatch --export=dataset=$dataset,pdb_type=AlphaFold2,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_AlphaFold2 script/slurm/train_woss.slurm
            
            sbatch --export=dataset=$dataset,pdb_type=ESMFold,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_ESMFold script/slurm/train.slurm
            sbatch --export=dataset=$dataset,pdb_type=ESMFold,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_ESMFold script/slurm/train_wofsss.slurm
            sbatch --export=dataset=$dataset,pdb_type=ESMFold,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_ESMFold script/slurm/train_wofs.slurm
            sbatch --export=dataset=$dataset,pdb_type=ESMFold,pooling_method=$pooling_method,plm_model=$plm_model --job-name="$dataset"_"$pooling_method"_ESMFold script/slurm/train_woss.slurm
        done
    done
done

for i in {8984..9000}
do
    scancel $i
done

############################################################################################################

# ablation for DeepLocBinary
for pooling_method in mean attention1d light_attention
do
    sbatch --export=dataset=DeepLocBinary,pooling_method=$pooling_method --job-name=deeploc-1_binary_"$pooling_method"_ESMFold script/slurm/train.slurm
    sbatch --export=dataset=DeepLocBinary,pooling_method=$pooling_method --job-name=deeploc-1_binary_"$pooling_method"_ESMFold script/slurm/train_wofsss.slurm
    sbatch --export=dataset=DeepLocBinary,pooling_method=$pooling_method --job-name=deeploc-1_binary_"$pooling_method"_ESMFold script/slurm/train_wofs.slurm
    sbatch --export=dataset=DeepLocBinary,pooling_method=$pooling_method --job-name=deeploc-1_binary_"$pooling_method"_ESMFold script/slurm/train_woss.slurm
done

############################################################################################################

# ablation for head number and pooling dropout
for head_num in 2 4 8 16 32
do
    for pooling_dropout in 0 0.1 0.25 0.5
    do
        sbatch --export=dataset=DeepLocBinary,num_heads=$head_num,pooling_dropout=$pooling_dropout --job-name=deeploc-1_binary_attention1d_"$head_num"_"$pooling_dropout"_ESMFold script/slurm/train_head.slurm
        sbatch --export=dataset=DeepLocMulti,num_heads=$head_num,pooling_dropout=$pooling_dropout --job-name=deeploc-1_multi_attention1d_"$head_num"_"$pooling_dropout"_ESMFold script/slurm/train_head.slurm
        sbatch --export=dataset=DeepSol,num_heads=$head_num,pooling_dropout=$pooling_dropout --job-name=deepsol_attention1d_"$head_num"_"$pooling_dropout"_ESMFold script/slurm/train_head.slurm
    done
done