#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=512M
#SBATCH --time=14-00:00:00
#SBATCH -o ./logs/slurm_output_%j_%x.out

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_LOCALID: $SLURM_LOCALID"

#source /home/$USER/.bashrc
#conda init
#conda activate xlstm_pretrained
ulimit -n
ulimit -n 16384
ulimit -n

export TORCH_CUDA_ARCH_LIST="9.0"

# run script from above
srun ~/.conda/envs/xlstm_pretrained/bin/python -u $1 --config_file $2