#!/bin/bash
python - << 'PY'

import torch

print("torch:", torch.__version__)

print("torch.version.cuda:", torch.version.cuda)

print("cuda available:", torch.cuda.is_available())

print("device_count:", torch.cuda.device_count())

PY

export HF_HUB_ETAG_TIMEOUT=20

export HF_HUB_READ_TIMEOUT=20

export HF_HUB_DOWNLOAD_TIMEOUT=20


export CUDA_VISIBLE_DEVICES=0,1,5,7	#4,5,6,7#0,1,2,3
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH: $PATH"



accelerate launch --mixed_precision bf16 --gpu_ids all --num_processes 4 train_fineweb.py > /home/ubuntu/zoe/TTFS_GPT2/out_xl.txt
