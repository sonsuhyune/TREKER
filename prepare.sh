#!/bin/bash

stage="Prepare"
dataset="R_test_all"
#dataset="mbert"

transformer="monologg/kobert" # "roberta-large"

max_seq_length=512
#filename="final_mbert"
filename="R_test_all"

python3 prepare.py --stage=${stage} --filename=${filename} --dataset=${dataset} --transformer=${transformer} --max_seq_length=${max_seq_length}