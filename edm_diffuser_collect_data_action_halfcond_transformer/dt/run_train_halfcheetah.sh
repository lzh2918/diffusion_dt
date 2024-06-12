#!/bin/bash

project="diffusion_dt"
group="the_original"
name="ho_20_scale_0.7_1.3"

task=("halfcheetah-medium-v2" "halfcheetah-medium-v2" "halfcheetah-medium-v2")
horizon=20
generate_percentage=(0.2 0.5 0.8)
target_returns=(5000.0 7000.0 9000.0)
train_file="/data/user/liuzhihong/paper/big_model/diffusion/decision_diffuser_collect_data/dt/dt.py"

i=0
log_dir="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/log/dt_half_$i"
data_path="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/first_edition/halfcheetah-medium-v2/horizon_20/24-0521-182906/halfcheetah-medium-v2/24-0522-192027/save_traj.npy"
name="ho_20_scale_0.7_1.3_per0.2"
CUDA_VISIBLE_DEVICES="5" python -u $train_file --project $project --generate_percentage ${generate_percentage[$i]}  --group $group --name $name --env_name ${task[$i]} --horizon $horizon --diffusion_data_load_path $data_path > $log_dir 2>&1 &

sleep 5 
i=1
log_dir="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/log/dt_half_$i"
data_path="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/first_edition/halfcheetah-medium-v2/horizon_20/24-0521-182906/halfcheetah-medium-v2/24-0522-192027/save_traj.npy"
name="ho_20_scale_0.7_1.3_per0.5"
CUDA_VISIBLE_DEVICES="5" python -u $train_file --project $project --generate_percentage ${generate_percentage[$i]}  --group $group --name $name --env_name ${task[$i]} --horizon $horizon --diffusion_data_load_path $data_path > $log_dir 2>&1 &

sleep 5
i=2
log_dir="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/log/dt_half_$i"
data_path="/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/first_edition/halfcheetah-medium-v2/horizon_20/24-0521-182906/halfcheetah-medium-v2/24-0522-192027/save_traj.npy"
name="ho_20_scale_0.7_1.3_per0.8"
CUDA_VISIBLE_DEVICES="5" python -u $train_file --project $project --generate_percentage ${generate_percentage[$i]}  --group $group --name $name --env_name ${task[$i]} --horizon $horizon --diffusion_data_load_path $data_path > $log_dir 2>&1 &

