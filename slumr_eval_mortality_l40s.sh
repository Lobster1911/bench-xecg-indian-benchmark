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

# run script from above
srun ~/.conda/envs/xlstm_pretrained/bin/python -u $1 --run_ids $2 $3 $4 $5 $6 --batch_size $7 --max_length_signal $8 --data_folder_music $9 --num_workers 64