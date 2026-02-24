# Lab tests

## ecg founder OK

for i in {1..5}; do
    sbatch --job-name=fm_ft_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/ecgfm_ft.yaml 
done

for i in {1..5}; do
    sbatch --job-name=fm_lp_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/ecgfm_lp.yaml
done

## jepa OK

for i in {1..5}; do
    sbatch --job-name=jepa_lp_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/jepa_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=jepa_ft_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/jepa_ft.yaml
done

## st-mem OK

for i in {1..5}; do
    sbatch --job-name=stmem_lp_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/stmem_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=stmem_ft_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/stmem_ft.yaml
done

## transformer OK

for i in {1..5}; do
    sbatch --job-name=trans_lp_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/transformer_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=trans_ft_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/transformer_ft.yaml 
done

## xlstm OK

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/xlstm_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/xlstm_ft.yaml
done 

## supervised

for i in {1..5}; do
    sbatch --job-name=xlstm_sup_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/xlstm_supervised.yaml
done 

## xlstm CODE15

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/xlstm_code15_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/xlstm_code15_ft.yaml
done 


## ECG-CPC

for i in {1..5}; do
    sbatch --job-name=cpc_lp_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/ecgcpc_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=cpc_ft_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/ecgcpc_ft.yaml
done 


## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/xlstm_dinoecg_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_lab slurm_train_l40s.sh train_lab_mimic.py configs/lab/xlstm_dinoecg_ft.yaml
done 
