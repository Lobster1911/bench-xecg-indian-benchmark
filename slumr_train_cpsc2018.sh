#!/bin/bash -l
#SBATCH --job-name=pretrain_xlstm
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=512M
#SBATCH --time=1-00:00:00
#SBATCH -o ./logs/slurm_output_%j_%x.out # STDOUT

echo "Visible GPUs: "
echo $CUDA_VISIBLE_DEVICES

#source /home/$USER/.bashrc
#conda init
#conda activate xlstm_pretrained

# run script from above
srun ~/.conda/envs/xlstm_pretrained/bin/python -u train_cpsc2018.py