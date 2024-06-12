#!/bin/bash

cuda=("2" "2" "3" "3")

project="diffusion_dt"
group="half_cond_1"
# name="ho_20_scale_0.7_1.3"

task=("hopper-medium-v2" "hopper-medium-v2" "hopper-medium-v2" "hopper-medium-v2")
horizon=20
generate_percentage=(1.0 0.2 0.5 0.0)
dataset_scale="(1.0_1.2)"
target_returns="(0.0_3000.0_4000.0)"
return_change_coef=1.0
train_file="/data/user/liuzhihong/paper/big_model/diffusion/decision_diffuser_collect_data_action_halfcond/dt/dt.py"
data_path="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/half_cond_1/hopper-medium-v2/horizon_20/24-0527-103632/hopper-medium-v2/24-0529-1146021.0_1.2/save_traj.npy"

$i

for ((i=0;i<=3;i++))
do
log_dir="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/log/halfcond_dt_$i.txt"
# data_path="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/first_edition/hopper-medium-v2/horizon_20/24-0521-182911/hopper-medium-v2/24-0524-120358/save_traj.npy"
CUDA_VISIBLE_DEVICES=${cuda[$i]} python -u $train_file --project $project --target_returns $target_returns --return_change_coef $return_change_coef --generate_percentage ${generate_percentage[$i]}  --group $group --env_name ${task[$i]} --horizon $horizon --dataset_scale $dataset_scale --diffusion_data_load_path $data_path > $log_dir 2>&1 &
sleep 2
done

