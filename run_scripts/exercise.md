# High intensity exercise

## ecg founder OK

for i in {1..4}; do
    sbatch --job-name=fm_ft_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/ecgfm_ft.yaml 
done

for i in {1..4}; do
    sbatch --job-name=fm_lp_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/ecgfm_lp.yaml
done

## jepa OK

for i in {1..4}; do
    sbatch --job-name=jepa_lp_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/jepa_lp.yaml 
done

for i in {1..4}; do
    sbatch --job-name=jepa_ft_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/jepa_ft.yaml
done

## st-mem OK

for i in {1..4}; do
    sbatch --job-name=stmem_lp_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/stmem_lp.yaml
done

for i in {1..4}; do
    sbatch --job-name=stmem_ft_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/stmem_ft.yaml
done

## transformer OK

for i in {1..4}; do
    sbatch --job-name=trans_lp_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/transformer_lp.yaml
done

for i in {1..4}; do
    sbatch --job-name=trans_ft_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/transformer_ft.yaml 
done

## xlstm OK

for i in {1..4}; do
    sbatch --job-name=xlstm_lp_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/xlstm_lp.yaml 
done

for i in {1..4}; do
    sbatch --job-name=xlstm_ft_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/xlstm_ft.yaml
done 

## supervised


for i in {1..5}; do
    sbatch --job-name=xlstm_sup_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/xlstm_supervised.yaml
done 


## xlstm CODE15

for i in {1..4}; do
    sbatch --job-name=xlstm_lp_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/xlstm_code15_lp.yaml 
done

for i in {1..4}; do
    sbatch --job-name=xlstm_ft_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/xlstm_code15_ft.yaml
done 

## ECG-CPC

for i in {1..5}; do
    sbatch --job-name=cpc_lp_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/ecgcpc_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=cpc_ft_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/ecgcpc_ft.yaml
done 


## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/xlstm_dinoecg_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_exe slurm_train_l40s.sh train_r_peak_intense.py configs/exercise/xlstm_dinoecg_ft.yaml
done 
