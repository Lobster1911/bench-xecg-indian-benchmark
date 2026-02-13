# MIT-BIH classification

## ecg founder OK

for i in {1..4}; do
    sbatch --job-name=fm_ft_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/ecgfm_ft.yaml 
done

for i in {1..4}; do
    sbatch --job-name=fm_lp_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/ecgfm_lp.yaml
done

## jepa OK

for i in {1..4}; do
    sbatch --job-name=jepa_lp_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/jepa_lp.yaml 
done 

for i in {1..4}; do
    sbatch --job-name=jepa_ft_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/jepa_ft.yaml
done

## st-mem OK

for i in {1..4}; do
    sbatch --job-name=stmem_lp_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/stmem_lp.yaml
done

for i in {1..4}; do
    sbatch --job-name=stmem_ft_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/stmem_ft.yaml
done

## transformer OK

for i in {1..4}; do
    sbatch --job-name=trans_lp_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/transformer_lp.yaml
done

for i in {1..4}; do
    sbatch --job-name=trans_ft_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/transformer_ft.yaml 
done

## xlstm OK

for i in {1..4}; do
    sbatch --job-name=xlstm_lp_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/xlstm_lp.yaml 
done

for i in {1..4}; do
    sbatch --job-name=xlstm_ft_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/xlstm_ft.yaml
done 

## supervised

for i in {1..5}; do
    sbatch --job-name=xlstm_sup_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/xlstm_supervised.yaml
done 

## xlstm CODE15

for i in {1..4}; do
    sbatch --job-name=xlstm_lp_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/xlstm_code15_lp.yaml 
done

for i in {1..4}; do
    sbatch --job-name=xlstm_ft_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/xlstm_code15_ft.yaml
done

## ECG-CPC

for i in {1..5}; do
    sbatch --job-name=cpc_lp_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/ecgcpc_lp.yaml 
done

for i in {1..4}; do
    sbatch --job-name=cpc_ft_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/ecgcpc_lp.yaml
done


## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/xlstm_dinoecg_lp.yaml 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_mit slurm_train_l40s.sh train_mit_bih.py configs/mit_bih_cls/xlstm_dinoecg_ft.yaml
done 
