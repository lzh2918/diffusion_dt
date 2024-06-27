#!/bin/bash

project="dt"
group="dt_upertimefix_0625_total"
# name="ho_20_scale_0.7_1.3"

task=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
horizon=20
generate_percentage_list="(0.0_0.5_1.0)"
cond_length=(2 5 10) 
eval_every=10000
diffusion_data_load_path=(/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/halfcheetah-medium-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0606-225729_upervf_date_24-0618-221347/24-0625-204304/save_traj.npy # halfcheetah medium 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/halfcheetah-medium-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0606-225727_upervf_date_24-0618-221351/24-0625-204306/save_traj.npy # halfcheetah medium 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/halfcheetah-medium-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0606-225725_upervf_date_24-0618-221355/24-0625-204308/save_traj.npy # halfcheetah medium 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/halfcheetah-medium-replay-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221400/24-0625-204310/save_traj.npy # halfcheetah replay 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/halfcheetah-medium-replay-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221403/24-0625-204312/save_traj.npy # halfcheetah replay 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/halfcheetah-medium-replay-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221408/24-0625-204314/save_traj.npy # halfcheetah replay 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/halfcheetah-medium-expert-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221414/24-0625-204316/save_traj.npy # halfcheetah expert 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/halfcheetah-medium-expert-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221422/24-0625-204318/save_traj.npy # halfcheetah expert 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/halfcheetah-medium-expert-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221425/24-0625-204320/save_traj.npy # halfcheetah expert 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/hopper-medium-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0606-225735_upervf_date_24-0618-221311/24-0625-204322/save_traj.npy # hopper medium 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/hopper-medium-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0606-225733_upervf_date_24-0618-221315/24-0625-204324/save_traj.npy # hopper medium 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/hopper-medium-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0606-225731_upervf_date_24-0618-221319/24-0625-204326/save_traj.npy # hopper medium 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/hopper-medium-replay-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221323/24-0625-204328/save_traj.npy  # hopper medium replay 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/hopper-medium-replay-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221327/24-0625-204330/save_traj.npy  # hopper medium replay 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/hopper-medium-replay-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221332/24-0625-204332/save_traj.npy # hopper medium replay 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/hopper-medium-expert-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221335/24-0625-204334/save_traj.npy # hopper medium expert 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/hopper-medium-expert-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221339/24-0625-204336/save_traj.npy # hopper medium expert 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/hopper-medium-expert-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221343/24-0625-204338/save_traj.npy # hopper medium expert 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/walker2d-medium-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0618-143025_upervf_date_24-0618-221427/24-0625-204341/save_traj.npy # walker2d medium 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/walker2d-medium-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0618-143025_upervf_date_24-0618-221428/24-0625-204342/save_traj.npy # walker2d medium 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/walker2d-medium-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0618-143025_upervf_date_24-0618-221433/24-0625-204345/save_traj.npy # walker2d medium 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/walker2d-medium-replay-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221436/24-0625-204346/save_traj.npy # walker2d medium replay 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/walker2d-medium-replay-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221440/24-0625-204348/save_traj.npy # walker2d medium replay 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/walker2d-medium-replay-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221445/24-0625-204350/save_traj.npy # walker2d medium replay 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/walker2d-medium-expert-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221450/24-0625-204352/save_traj.npy # walker2d medium expert 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/walker2d-medium-expert-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221455/24-0625-204354/save_traj.npy # walker2d medium expert 5
                          //home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/fixtime_upervf/walker2d-medium-expert-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-221500/24-0625-204356/save_traj.npy # walker2d medium expert 10
                          )
upervf_path=(/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221347/uper_value_func_checkpoint.pt # halfcheetah medium 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221351/uper_value_func_checkpoint.pt # halfcheetah medium 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221355/uper_value_func_checkpoint.pt # halfcheetah medium 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221400/uper_value_func_checkpoint.pt # halfcheetah medium replay 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221403/uper_value_func_checkpoint.pt # halfcheetah medium replay 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221408/uper_value_func_checkpoint.pt # halfcheetah medium replay 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221414/uper_value_func_checkpoint.pt # halfcheetah medium expert 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221422/uper_value_func_checkpoint.pt # halfcheetah medium expert 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221425/uper_value_func_checkpoint.pt # halfcheetah medium expert 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221311/uper_value_func_checkpoint.pt # hopper medium 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221315/uper_value_func_checkpoint.pt # hopper medium 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221319/uper_value_func_checkpoint.pt # hopper medium 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221323/uper_value_func_checkpoint.pt # hopper medium replay 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221327/uper_value_func_checkpoint.pt # hopper medium replay 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221332/uper_value_func_checkpoint.pt # hopper medium replay 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221335/uper_value_func_checkpoint.pt # hopper medium expert 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221339/uper_value_func_checkpoint.pt # hopper medium expert 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221343/uper_value_func_checkpoint.pt # hopper medium expert 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221427/uper_value_func_checkpoint.pt # walker2d medium 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221428/uper_value_func_checkpoint.pt # walker2d medium 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221433/uper_value_func_checkpoint.pt # walker2d medium 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221436/uper_value_func_checkpoint.pt # walker2d medium replay 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221440/uper_value_func_checkpoint.pt # walker2d medium replay 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221445/uper_value_func_checkpoint.pt # walker2d medium replay 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221450/uper_value_func_checkpoint.pt # walker2d medium expert 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221455/uper_value_func_checkpoint.pt # walker2d medium expert 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221500/uper_value_func_checkpoint.pt # walker2d medium expert 10
             )

train_file="/home/liuzhihong/diffusion_related/diffusion_dt/code/decision_diffuser_collect_data_action_halfcond_transformer/dt/batch_dt_upervf_uper.py"
model_index_map=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26)
cuda_index_map=(1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7)
env_map=(0 1 2 3 4 5 6 7 8)

$i
$j
$index
$model_index
$cuda_index
$env_index

for ((i=3;i<4;i++)) # 4个环境
do
    for ((j=2;j<3;j++)) # cond length 3
    do
        ((index= i*3 + j))
        ((model_index = ${model_index_map[$index]}))
        ((cuda_index = ${cuda_index_map[$index]}))
        ((env_index = ${env_map[$i]}))
        echo $index
        log_dir="/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/log/collect_data/log/halfcond_dt_$index.txt"
        CUDA_VISIBLE_DEVICES=$cuda_index python -u $train_file\
                                                --project $project \
                                                --group $group \
                                                --env_name ${task[$env_index]} \
                                                --eval_every $eval_every \
                                                --horizon $horizon \
                                                --generate_percentage_list $generate_percentage_list  \
                                                --cond_length ${cond_length[$j]}\
                                                --diffusion_data_load_path ${diffusion_data_load_path[$model_index]}\
                                                --uper_vf_path ${upervf_path[$model_index]} > $log_dir 2>&1 &
        sleep 2
    done
done

