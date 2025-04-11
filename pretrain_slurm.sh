#!/bin/bash -l

#SBATCH --job-name=pretrain_xlstm
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1G
#SBATCH --time=2-00:00:00
#SBATCH -o ./logs/slurm_output_%j_%x.out # STDOUT

echo "Visible GPUs: "
echo $CUDA_VISIBLE_DEVICES

source /home/$USER/.bashrc
conda init
conda activate xlstm_pretrained

# run script from above
srun python3 -u pretrain.py