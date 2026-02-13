# Sleep apnea

## ecg founder OK

for i in {1..4}; do
    sbatch --job-name=fm_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgfm_ft.yaml 
done

for i in {1..4}; do
    sbatch --job-name=fm_lp_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgfm_lp.yaml
done

## jepa OK

for i in {1..4}; do
    sbatch --job-name=jepa_lp_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/jepa_lp.yaml 
done

for i in {1..4}; do
    sbatch --job-name=jepa_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/jepa_ft.yaml
done

## st-mem OK

for i in {1..4}; do
    sbatch --job-name=stmem_lp_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/stmem_lp.yaml
done

for i in {1..4}; do
    sbatch --job-name=stmem_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/stmem_ft.yaml
done

## transformer OK

for i in {1..4}; do
    sbatch --job-name=trans_lp_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/transformer_lp.yaml
done

for i in {1..4}; do
    sbatch --job-name=trans_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/transformer_ft.yaml 
done

## xlstm OK

for i in {1..4}; do
    sbatch --job-name=xlstm_lp_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/xlstm_lp.yaml 
done

for i in {1..4}; do
    sbatch --job-name=xlstm_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/xlstm_ft.yaml
done 

## supervised

for i in {1..5}; do
    sbatch --job-name=xlstm_sup_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/xlstm_supervised.yaml
done 


## xlstm CODE15

for i in {1..4}; do
    sbatch --job-name=xlstm_lp_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/xlstm_code15_lp.yaml 
done

for i in {1..4}; do
    sbatch --job-name=xlstm_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/xlstm_code15_ft.yaml
done 


## ECG-CPC

for i in {1..4}; do
    sbatch --job-name=xlstm_lp_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgcpc_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgcpc_ft_10s.yaml
done 

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgcpc_ft_20s.yaml
done 

for i in {1..4}; do
    sbatch --job-name=xlstm_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgcpc_ft_30s.yaml
done 

for i in {1..4}; do
    sbatch --job-name=xlstm_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgcpc_ft_1m.yaml
done 

for i in {1..4}; do
    sbatch --job-name=xlstm_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgcpc_ft_3m.yaml
done 

for i in {1..4}; do
    sbatch --job-name=cpc_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgcpc_ft_5m.yaml
done 

for i in {1..5}; do
    sbatch --job-name=cpc_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/ecgcpc_ft_9m.yaml
done 

## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/xlstm_dinoecg_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_sa slurm_train_l40s.sh train_sleep_apnea.py configs/sleep_apnea/xlstm_dinoecg_ft.yaml
done 
