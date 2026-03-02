#!/bin/bash -l
#SBATCH --job-name=train_model
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
#ulimit -n 16384

ulimit -n
ulimit -n 16384 
ulimit -n

export PYKEOPS_VERBOSE=1
export KEOPS_RECOMPILE=1

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="8.9"

SCRIPT=$1
CONFIG=$2
shift 2

srun uv run "$SCRIPT" --config_file "$CONFIG" "$@"