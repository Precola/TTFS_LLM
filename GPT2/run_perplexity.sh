export CUDA_VISIBLE_DEVICES=5
python perplexity_xl.py
#$accelerate launch --mixed_precision bf16 perplexity.py > /home/ubuntu/zoe/TTFS_GPT2/logs/out_perplexity.txt
