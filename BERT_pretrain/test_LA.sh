#!/bin/bash


source ~/zoe_env/bin/activate
cd /home/zoe/TTFS-BERT/Tasks/


python TaskForLAPretraining_TTFS.py >> output_LA.txt 2>&1 #&# LN changed gelu fixed
