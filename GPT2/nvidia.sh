#!/bin/bash
#SBATCH --job-name="GLUE"
#SBATCH --output=output_base%j.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:2               # 2 GPU
#SBATCH --time=20-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/env/bin/activate
#export CUDA_VISIBLE_DEVICES=1
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH: $PATH"
echo "=== NVIDIA-SMI ==="
nvidia-smi
echo "=== NVIDIA-SMI ==="