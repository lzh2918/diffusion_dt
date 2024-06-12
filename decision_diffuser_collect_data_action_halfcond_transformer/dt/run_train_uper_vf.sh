#!/bin/bash

cuda=("0" "3" "4")

project="dt"
group="dt_uper_0612base"
# name="ho_20_scale_0.7_1.3"

task=("hopper-medium-v2")
horizon=20
generate_percentage=(1.0 0.2 0.6 0.0)
target_returns="(0.0_3000.0_5000.0)"
cond_length=(10 5 2)
diffusion_data_load_path=(/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/store_data/hopper-medium-v2/diffusion_horizon_20_cond_length_1024-0606-225731/24-0610-214516_er_0.95_cond_length_10/save_traj.npy
                     /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/store_data/hopper-medium-v2/diffusion_horizon_20_cond_length_524-0606-225733/24-0610-214535_er_0.95_cond_length_5/save_traj.npy
                     /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/store_data/hopper-medium-v2/diffusion_horizon_20_cond_length_224-0606-225735/24-0610-214534_er_0.95_cond_length_2/save_traj.npy
                    )
uper_vf_path=(/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_10/24-0611-161800/uper_value_func_checkpoint.pt
            /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_5/24-0611-161752/uper_value_func_checkpoint.pt
            /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_2/24-0611-161744/uper_value_func_checkpoint.pt
           )

train_file="/home/liuzhihong/diffusion_related/diffusion_dt/code/decision_diffuser_collect_data_action_halfcond_transformer/dt/dt_upervf.py"

$i
$j
k=0
$index

for ((i=0;i<3;i++))
do
    for ((j=0;j<4;j++))
    do
        ((index= 4*i+j))
        log_dir="/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/log/collect_data/log/halfcond_dt_$index.txt"
        CUDA_VISIBLE_DEVICES=${cuda[$i]} python -u $train_file\
                                                --project $project \
                                                --group $group \
                                                --env_name ${task[$k]} \
                                                --horizon $horizon \
                                                --generate_percentage ${generate_percentage[$j]}  \
                                                --target_returns $target_returns\
                                                --cond_length ${cond_length[$i]}\
                                                --diffusion_data_load_path ${diffusion_data_load_path[$i]}\
                                                --uper_vf_path ${uper_vf_path[$i]} > $log_dir 2>&1 &
        sleep 2
    done
done

