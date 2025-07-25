#!/bin/bash -l
#SBATCH --job-name=pretrain
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=512M
#SBATCH --time=7-00:00:00
#SBATCH -o ./logs/slurm_output_%j_%x.out # STDOUT

echo "Visible GPUs: "
echo $CUDA_VISIBLE_DEVICES

#source /home/$USER/.bashrc
#conda init
#conda activate xlstm_pretrained

ulimit -n
ulimit -n 16384
ulimit -n

# run script from above
srun ~/.conda/envs/xlstm_pretrained/bin/python -u pretrain.py