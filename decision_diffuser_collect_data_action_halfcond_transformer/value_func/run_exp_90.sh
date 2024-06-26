#!/bin/bash

cuda=("0" "1" "2" "3")

project="uper_value_func_0623"
group="halfcond_transformer_noreward"

task=("walker2d-medium-v2" "walker2d-medium-expert-v2" "walker2d-medium-expert-v2")
cond_length=(2 5 10)
er_coef=(0.8 0.95)
horizon=20
num_layers=4
num_heads=2
save_checkpoints=True
train_file="/home/liuzhihong/diffusion_dt_temp/code/decision_diffuser_collect_data_action_halfcond_transformer/value_func/uper_value_function.py"

$i
$j
$k
$index
$t_index
$temp_cuda # 程序楼开了一个，所以在每张卡上单独开一个，设置的变量

for ((i=0;i<3;i++))
do
    for ((j=0;j<3;j++))
    do
        for ((k=0;k<2;k++))
        do
            ((t_index = 6*i + 2*j + k))
            ((temp_cuda = 2*j + k))
            log_dir="/home/liuzhihong/diffusion_dt_temp/exp_result/log/uper_value_func_$t_index.txt"
            CUDA_VISIBLE_DEVICES=${cuda[$i]} python -u $train_file \
                                                    --project $project \
                                                    --group $group \
                                                    --env_name ${task[$i]} \
                                                    --cond_length ${cond_length[$j]} \
                                                    --num_layers $num_layers\
                                                    --num_heads $num_heads\
                                                    --er_coef ${er_coef[$k]} \
                                                    --horizon $horizon \
                                                    --save_checkpoints $save_checkpoints > $log_dir 2>&1 &
            sleep 2
        done
    done 
done

