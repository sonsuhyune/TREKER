#!/bin/bash

stage="Main"
dataset="for_test_base" ### 아래 weight 수정 해야함!!!!!!!!!! R_test_all

seed=7894


transformer="klue/bert-base" # "roberta-large"


hidden_size=768
bilinear_block_size=64

RE_max=4
CR_focal_gamma=2
PER_focal_gamma=2
FER_threshold=0.5

loss_weight_CR=0.1
loss_weight_ET=0.1
loss_weight_PER=0
loss_weight_FER=0

num_epoch=30
batch_size=1
update_freq=1

new_lr=1e-4
pretrained_lr=5e-5
warmup_ratio=0.06
max_grad_norm=1.0

save_name = "for_test_base"

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=47041 main.py  --stage=${stage} --save_name = ${save_name} --dataset=${dataset} --seed=${seed} --transformer=${transformer} --hidden_size=${hidden_size} --bilinear_block_size=${bilinear_block_size} --RE_max=${RE_max} --CR_focal_gamma=${CR_focal_gamma} --PER_focal_gamma=${PER_focal_gamma} --FER_threshold=${FER_threshold} --loss_weight_CR=${loss_weight_CR} --loss_weight_ET=${loss_weight_ET} --loss_weight_PER=${loss_weight_PER} --loss_weight_FER=${loss_weight_FER} --num_epoch=${num_epoch} --batch_size=${batch_size} --update_freq=${update_freq} --new_lr=${new_lr} --pretrained_lr=${pretrained_lr} --warmup_ratio=${warmup_ratio} --max_grad_norm=${max_grad_norm}