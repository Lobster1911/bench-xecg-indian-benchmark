# Deepbeat PPG

## ecg founder OK

for i in {1..5}; do
    sbatch --job-name=fm_ft_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/ecgfm_ft.yaml 
done

for i in {1..5}; do
    sbatch --job-name=fm_lp_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/ecgfm_lp.yaml
done

## jepa OK

for i in {1..5}; do
    sbatch --job-name=jepa_lp_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/jepa_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=jepa_ft_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/jepa_ft.yaml
done

## st-mem OK

for i in {1..5}; do
    sbatch --job-name=stmem_lp_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/stmem_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=stmem_ft_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/stmem_ft.yaml
done

## transformer OK

for i in {1..5}; do
    sbatch --job-name=trans_lp_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/transformer_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=trans_ft_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/transformer_ft.yaml 
done

## xlstm OK

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/xlstm_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/xlstm_ft.yaml
done 


## supervised

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/xlstm_supervised.yaml
done 

## xlstm CODE15

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/xlstm_code15_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/xlstm_code15_ft.yaml
done 


## ECG-CPC

for i in {1..4}; do
    sbatch --job-name=cpc_lp_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/ecgcpc_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=cpc_ft_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/ecgcpc_ft.yaml
done 


## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/xlstm_dinoecg_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_db slurm_train_l40s.sh train_deepbeat.py configs/deepbeat/xlstm_dinoecg_ft.yaml
done 
