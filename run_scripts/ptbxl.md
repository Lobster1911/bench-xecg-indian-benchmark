# PTB XL

## ecg founder OK

for i in {1..5}; do
    sbatch --job-name=fm_ft_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/ecgfm_ft.yaml 
done

for i in {1..5}; do
    sbatch --job-name=fm_lp_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/ecgfm_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=fm_lp10_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/ecgfm_10pct.yaml 
done

for i in {1..5}; do
    sbatch --job-name=fm_lp1_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/ecgfm_1pct.yaml
done


## jepa OK

for i in {1..5}; do
    sbatch --job-name=jepa_lp1_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/jepa_1pct.yaml 
done

for i in {1..5}; do
    sbatch --job-name=jepa_lp10_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/jepa_10pct.yaml
done

for i in {1..5}; do
    sbatch --job-name=jepa_lp_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/jepa_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=jepa_ft_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/jepa_ft.yaml
done

## st-mem OK

for i in {1..5}; do
    sbatch --job-name=stmem_lp1_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/stmem_1pct.yaml 
done

for i in {1..5}; do
    sbatch --job-name=stmem_lp10_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/stmem_10pct.yaml 
done

for i in {1..5}; do
    sbatch --job-name=stmem_lp_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/stmem_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=stmem_ft_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/stmem_ft.yaml
done

## transformer OK

for i in {1..5}; do
    sbatch --job-name=trans_lp_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/transformer_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=trans_lp1_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/transformer_lp_1pct.yaml
done

for i in {1..5}; do
    sbatch --job-name=trans_lp10_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/transformer_lp_10pct.yaml
done

for i in {1..5}; do
    sbatch --job-name=trans_ft_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/transformer_ft.yaml 
done

## xlstm OK

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/xlstm_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_lp1_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/xlstm_lp_1pct.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_lp10_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/xlstm_lp_10pct.yaml
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/xlstm_ft.yaml
done 

## supervised

for i in {1..5}; do
    sbatch --job-name=xlstm_sup_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/xlstm_supervised.yaml
done 

## xlstm code15

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/xlstm_code15_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/xlstm_code15_ft.yaml 
done


## ECG CPC

for i in {1..5}; do
    sbatch --job-name=cpc_lp_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/ecgcpc_lp.yaml
done

for i in {1..5}; do
    sbatch --job-name=cpc_ft_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/ecgcpc_ft.yaml 
done


## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/xlstm_dinoecg_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_ptb-xl slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl/xlstm_dinoecg_ft.yaml
done 

## xlstm HEEDB

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_ptb-xl slurm_train_h100.sh train_ptb_xl.py configs/ptb-xl/xlstm_heedb_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_ptb-xl slurm_train_h100.sh train_ptb_xl.py configs/ptb-xl/xlstm_heedb_ft.yaml
done 
