# Mortality

## ecg founder OK

for i in {1..5}; do
    sbatch --job-name=fm_ft_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/ecgfm_ft.yaml 
done

for i in {1..5}; do
    sbatch --job-name=fm_lp_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/ecgfm_lp.yaml
done

## jepa OK

for i in {1..5}; do
    sbatch --job-name=jepa_lp_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/jepa_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=jepa_ft_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/jepa_ft.yaml
done

## st-mem OK

for i in {1..5}; do
    sbatch --job-name=stmem_lp_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/stmem_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=stmem_ft_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/stmem_ft.yaml
done

## transformer OK

for i in {1..5}; do
    sbatch --job-name=trans_lp_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/transformer_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=trans_ft_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/transformer_ft.yaml 
done

## xlstm OK

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/xlstm_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/xlstm_ft.yaml
done 

## supervised

for i in {1..5}; do
    sbatch --job-name=xlstm_sup_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/xlstm_supervised.yaml
done 

## xlstm CODE15

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/xlstm_code15_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/xlstm_code15_ft.yaml
done 

## ECG-CPC
for i in {1..5}; do
    sbatch --job-name=cpc_lp_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/ecgcpc_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=cpc_ft_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/ecgcpc_ft.yaml
done 


## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/xlstm_dinoecg_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_mortality slurm_train_l40s.sh train_mortality.py configs/mortality/xlstm_dinoecg_ft.yaml
done 
