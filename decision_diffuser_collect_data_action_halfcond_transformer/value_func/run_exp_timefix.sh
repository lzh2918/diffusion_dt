#!/bin/bash

cuda=("0" "0" "3" "4" "7" "4")

project="uper_value_func"
group="halfcond_transformer_noreward"

task=("hopper-medium-replay-v2" "halfcheetah-medium-expert-v2" "halfcheetah-medium-replay-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
cond_length=(2 5 10)
er_coef=(0.5 0.95)
horizon=20
save_checkpoints=True
train_file="/home/liuzhihong/diffusion_related/diffusion_dt/code/decision_diffuser_collect_data_action_halfcond_transformer/value_func/uper_value_function_fixtime.py"

$i
$j
$k
$index
$t_index
$temp_cuda # 程序楼开了一个，所以在每张卡上单独开一个，设置的变量

for ((i=1;i<2;i++))
do
    for ((j=2;j<3;j++))
    do
        for ((k=1;k<2;k++))
        do
            ((t_index = 6*i + 2*j + k))
            ((temp_cuda = 2*j + k))
            log_dir="/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/log/collect_data/log/uper_value_func_$t_index.txt"
            CUDA_VISIBLE_DEVICES=${cuda[$temp_cuda]} python -u $train_file \
                                                    --project $project \
                                                    --group $group \
                                                    --env_name ${task[$i]} \
                                                    --cond_length ${cond_length[$j]} \
                                                    --er_coef ${er_coef[$k]} \
                                                    --horizon $horizon \
                                                    --save_checkpoints $save_checkpoints > $log_dir 2>&1 &
            sleep 2
        done
    done 
done

