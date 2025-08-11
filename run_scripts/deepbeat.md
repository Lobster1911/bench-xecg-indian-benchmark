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