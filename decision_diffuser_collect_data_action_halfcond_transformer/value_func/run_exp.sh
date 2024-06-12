#!/bin/bash

cuda=("3" "3" "3" "4" "4" "4" "5" "5" "5" "6" "6" "6")

project="uper_value_func"
group="halfcond_transformer_noreward"
# name="ho_20_scale_0.7_1.3"

task=("hopper-medium-v2" "halfcheetah-medium-v2")
cond_length=(2 5 10)
er_coef=(0.5 0.75 0.9 0.95)
horizon=20
save_checkpoints=True
train_file="/home/liuzhihong/diffusion_related/diffusion_dt/decision_diffuser_collect_data_action_halfcond_transformer/value_func/uper_value_function.py"

$i
$j
$k
$index
$t_index

for ((i=0;i<=0;i++))
do
    for ((j=0;j<=2;j++))
    do
        for ((k=0;k<=3;k++))
        do
            ((t_index = 12*i + 4*j + k))
            log_dir="/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/collect_data/log/halfcond_dt_$t_index.txt"
            # data_path="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/first_edition/hopper-medium-v2/horizon_20/24-0521-182911/hopper-medium-v2/24-0524-120358/save_traj.npy"
            # ((index = 3*i + j))
            CUDA_VISIBLE_DEVICES=${cuda[$t_index]} python -u $train_file --project $project --group $group --env_name ${task[$i]} --horizon $horizon --cond_length ${cond_length[$j]} --er_coef ${er_coef[$k]} --save_checkpoints $save_checkpoints > $log_dir 2>&1 &
            sleep 2
        done
    done
done

