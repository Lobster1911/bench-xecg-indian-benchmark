#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --time=14-00:00:00
#SBATCH -o ./logs/slurm_output_%j_%x.out # STDOUT

source ~/.bashrc
conda activate xlstm_pretrained

python -u lvef_parsing.py /home/datasets/mimic-iv-ecg/raw/discharge.csv.gz \
       --output_path /home/datasets/mimic-iv-ecg/raw/discharge_lvef.csv \
       --test