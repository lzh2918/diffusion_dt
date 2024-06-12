#!/bin/bash

group="half_cond_diffusion"
cuda=("2" "3" "4" "5" "6" "7")
# group="test"
task=("halfcheetah-medium-expert-v2" "hopper-medium-expert-v2" "walker2d-medium-expert-v2" "halfcheetah-medium-replay-v2" "hopper-medium-replay-v2" "walker2d-medium-replay-v2" )
train_file="/home/liuzhihong/diffusion_related/diffusion_dt/decision_diffuser_collect_data_action_halfcond_transformer/analysis/train.py"
horizon=20
cond_length=(10 5 2)


$i
$j
$log_index
$cuda_index

for ((i=0;i<=5;i++))
do
    for ((j=0;j<3;j++))
    do
    ((log_index = i*3+j))
    log_dir="/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/collect_data/log/half_cond_diffusion_$log_index.txt"
    CUDA_VISIBLE_DEVICES=${cuda[$i]} python -u $train_file --group $group --task ${task[$i]} --cond_length ${cond_length[$j]} --horizon $horizon > $log_dir 2>&1 &
    done
sleep 2
done


