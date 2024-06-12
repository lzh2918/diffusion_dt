#!/bin/bash

collect_num=50000
# collect_num=1000
horizon=20
task=("hopper-medium-v2" "halfcheetah-medium-v2")
file_path=/data/user/liuzhihong/paper/big_model/diffusion/decision_diffuser_collect_data_action_halfcond/analysis/eval_store.py

i=0
logfile=/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/log/collect_log_action_diffusion_$i.txt
loadpath=/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/half_cond_1/hopper-medium-v2/horizon_20/24-0527-103632/checkpoint
CUDA_VISIBLE_DEVICES="2" python -u $file_path --task ${task[$i]} --loadpath $loadpath --collect_num $collect_num --horizon $horizon > $logfile 2>&1 &

sleep 5

i=1
logfile=/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/log/collect_log_action_diffusion_$i.txt
loadpath=/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/half_cond_1/halfcheetah-medium-v2/horizon_20/24-0527-103627/checkpoint
CUDA_VISIBLE_DEVICES="3" python -u $file_path --task ${task[$i]} --loadpath $loadpath --collect_num $collect_num --horizon $horizon > $logfile 2>&1 &