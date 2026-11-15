#!/bin/bash


source ~/zoe_env/bin/activate
cd /home/zoe/TTFS-BERT/Tasks/
python TaskForSingleSentenceClassification_TTFS.py >> output_mnli.txt 2>&1
python TaskForSingleSentenceClassification_TTFS.py --seed 713 >> output_mnli.txt 2>&1
#python TaskForSingleSentenceClassification_TTFS.py --seed 1118 >> output_qnli.txt 2>&1