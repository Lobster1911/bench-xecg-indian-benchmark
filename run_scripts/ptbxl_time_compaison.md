# PTB XL


sbatch --job-name=fm_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/ecgfm_ft.yaml 

sbatch --job-name=fm_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/ecgfm_lp.yaml

sbatch --job-name=jepa_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/jepa_lp.yaml 

sbatch --job-name=jepa_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/jepa_ft.yaml

sbatch --job-name=stmem_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/stmem_lp.yaml

sbatch --job-name=stmem_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/stmem_ft.yaml

sbatch --job-name=trans_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/transformer_lp.yaml

sbatch --job-name=trans_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/transformer_ft.yaml 

sbatch --job-name=xlstm_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/xlstm_lp.yaml 

sbatch --job-name=xlstm_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/xlstm_ft.yaml

sbatch --job-name=xlstm_sup_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/xlstm_supervised.yaml

sbatch --job-name=cpc_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/ecgcpc_lp.yaml 

sbatch --job-name=cpc_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-comparison/ecgcpc_ft.yaml
