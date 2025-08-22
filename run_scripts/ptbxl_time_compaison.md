# PTB XL


sbatch --job-name=fm_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/ecgfm_ft.yaml 

sbatch --job-name=fm_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/ecgfm_lp.yaml

sbatch --job-name=jepa_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/jepa_lp.yaml 

sbatch --job-name=jepa_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/jepa_ft.yaml

sbatch --job-name=stmem_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/stmem_lp.yaml

sbatch --job-name=stmem_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/stmem_ft.yaml

sbatch --job-name=trans_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/transformer_lp.yaml

sbatch --job-name=trans_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/transformer_ft.yaml 

sbatch --job-name=xlstm_lp_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/xlstm_lp.yaml 

sbatch --job-name=xlstm_ft_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/xlstm_ft.yaml

sbatch --job-name=xlstm_sup_ptb slurm_train_l40s.sh train_ptb_xl.py configs/ptb-xl-time-compariosn/xlstm_supervised.yaml
