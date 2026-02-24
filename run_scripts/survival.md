# Mortality

## ecg founder OK

for i in {1..5}; do
    sbatch --job-name=fm_ft_survival slurm_train_l40s.sh train_survival.py configs/survival/ecgfm_ft.yaml 
done

for i in {1..5}; do
    sbatch --job-name=fm_lp_survival slurm_train_l40s.sh train_survival.py configs/survival/ecgfm_lp.yaml
done

## jepa OK

for i in {1..5}; do
    sbatch --job-name=jepa_lp_survival slurm_train_l40s.sh train_survival.py configs/survival/jepa_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=jepa_ft_survival slurm_train_l40s.sh train_survival.py configs/survival/jepa_ft.yaml
done

## st-mem OK

for i in {1..5}; do
    sbatch --job-name=stmem_lp_survival slurm_train_l40s.sh train_survival.py configs/survival/stmem_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=stmem_ft_survival slurm_train_l40s.sh train_survival.py configs/survival/stmem_ft.yaml
done

## transformer OK

for i in {1..5}; do
    sbatch --job-name=trans_lp_survival slurm_train_l40s.sh train_survival.py configs/survival/transformer_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=trans_ft_survival slurm_train_l40s.sh train_survival.py configs/survival/transformer_ft.yaml 
done

## xlstm OK

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_survival slurm_train_l40s.sh train_survival.py configs/survival/xlstm_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_survival slurm_train_l40s.sh train_survival.py configs/survival/xlstm_ft.yaml
done 

## supervised

for i in {1..5}; do
    sbatch --job-name=xlstm_sup_survival slurm_train_l40s.sh train_survival.py configs/survival/xlstm_supervised.yaml
done 

## xlstm CODE15

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_survival slurm_train_l40s.sh train_survival.py configs/survival/xlstm_code15_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_survival slurm_train_l40s.sh train_survival.py configs/survival/xlstm_code15_ft.yaml
done 

## ECG-CPC
for i in {1..5}; do
    sbatch --job-name=cpc_lp_survival slurm_train_l40s.sh train_survival.py configs/survival/ecgcpc_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=cpc_ft_survival slurm_train_l40s.sh train_survival.py configs/survival/ecgcpc_ft.yaml
done 


## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_survival slurm_train_l40s.sh train_survival.py configs/survival/xlstm_dinoecg_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_survival slurm_train_l40s.sh train_survival.py configs/survival/xlstm_dinoecg_ft.yaml
done 
