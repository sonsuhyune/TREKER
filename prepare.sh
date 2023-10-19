#!/bin/bash

stage="Prepare"
dataset="TREK_dataset"
#dataset="mbert"

transformer="klue/bert-base" # "roberta-large"

max_seq_length=512


python3 prepare.py --stage=${stage} --filename=${filename} --dataset=${dataset} --transformer=${transformer} --max_seq_length=${max_seq_length}