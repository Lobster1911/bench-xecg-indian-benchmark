#!/bin/bash -l
#SBATCH --job-name=pretrain
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1G
#SBATCH --time=14-00:00:00
#SBATCH -o ./logs/slurm_output_%j_%x.out # STDOUT

echo "Visible GPUs: "
echo $CUDA_VISIBLE_DEVICES

ulimit -n 16384

#source /home/$USER/.bashrc
#conda init
#conda activate xlstm_pretrained

# run script from above
srun ~/.conda/envs/xlstm_pretrained/bin/python -u pretrain.py