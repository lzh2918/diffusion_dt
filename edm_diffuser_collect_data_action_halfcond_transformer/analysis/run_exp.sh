#!/bin/bash
cuda=("2" "5" "4" "5" "6" "7")
group="edm_diffusion"
# group="test"
task=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "halfcheetah-medium-expert-v2" "hopper-medium-expert-v2" "walker2d-medium-expert-v2")
cond_length=(10 5 2) 
horizon=20
train_file="/home/liuzhihong/diffusion_related/diffusion_dt/code/edm_diffuser_collect_data_action_halfcond_transformer/analysis/train.py"


$i
$j
$log_index
$cuda_index

for ((i=1;i<=1;i++))
do
    for ((j=0;j<3;j++))
    do
    ((log_index = i*3+j))
    log_dir="/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/log/collect_data/log/edm_diffusion_$log_index.txt"
    CUDA_VISIBLE_DEVICES=${cuda[$i]} python -u $train_file \
                                            --group $group \
                                            --task ${task[$i]} \
                                            --cond_length ${cond_length[$j]} \
                                            --horizon $horizon > $log_dir 2>&1 &
    done
sleep 2
done


