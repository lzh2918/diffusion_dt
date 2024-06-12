#!/bin/bash

group="origin"

task=("halfcheetah-medium-v2" "hopper-medium-expert-v2")
train_file="/data/user/liuzhihong/paper/big_model/diffusion/decision_diffuser/analysis/train.py"

i=0
log_dir="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser/log/dd_$i"
CUDA_VISIBLE_DEVICES="5" python -u $train_file --group $group --task ${task[$i]} > $log_dir 2>&1 &

sleep 5 
i=1
log_dir="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser/log/dd_$i"
CUDA_VISIBLE_DEVICES="7" python -u $train_file --group $group --task ${task[$i]} > $log_dir 2>&1 &

