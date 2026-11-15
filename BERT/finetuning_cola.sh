#!/bin/bash


source ~/zoe_env/bin/activate
cd /home/zoe/TTFS-BERT/Tasks/
python TaskForSingleSentenceClassification_TTFS.py --task_name "cola">> output_cola.txt 2>&1
python TaskForSingleSentenceClassification_TTFS.py --seed 713 --task_name "cola">> output_cola.txt 2>&1
#python TaskForSingleSentenceClassification_TTFS.py --seed 1118 --task_name "cola">> output_qnli.txt 2>&1