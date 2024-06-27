#!/bin/bash

project="upervf_iql"
group="upervf_iql_0627"

task=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
iql_tau=(0.5 0.7 0.95)
discount_list="(1.0_0.99_0.995)"
save_checkpoints=True
train_file="/home/liuzhihong/diffusion_related/diffusion_dt/code/decision_diffuser_collect_data_action_halfcond_transformer/value_func/iqlvf_uper.py"

env_map=(0 1 2 3 4 5 6 7 8)
cuda_map=(0 0 0 0 0 1 1 1 1 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7)

$i
$j
$total_index
$env_index
$cuda_index # 程序楼开了一个，所以在每张卡上单独开一个，设置的变量

for ((i=0;i<9;i++))
do
    for ((j=0;j<3;j++))
    do
        ((total_index = 3*i + j))
        ((env_index = i))
        ((cuda_index = ${cuda_map[$total_index]}))
        echo $total_index
        log_dir="/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/log/collect_data/log/upervf_iql_$total_index.txt"
        CUDA_VISIBLE_DEVICES=$cuda_index python -u $train_file \
                                                --project $project \
                                                --group $group \
                                                --env ${task[$env_index]} \
                                                --iql_tau ${iql_tau[$j]} \
                                                --discount_list $discount_list \
                                                --save_checkpoints $save_checkpoints > $log_dir 2>&1 &
        sleep 2
        
    done 
done

