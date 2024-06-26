#!/bin/bash

project="dt"
group="dt_upertimelong_0625_total"
# name="ho_20_scale_0.7_1.3"

task=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
horizon=20
generate_percentage_list="(0.0_0.5_1.0)"
cond_length=(2 5 10) 
eval_every=10000
diffusion_data_load_path=(/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/halfcheetah-medium-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0606-225729_upervf_date_24-0613-133827/24-0620-140355/save_traj.npy # halfcheetah medium 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/halfcheetah-medium-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0606-225727_upervf_date_24-0613-133831/24-0620-140355/save_traj.npy # halfcheetah medium 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/halfcheetah-medium-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0606-225725_upervf_date_24-0613-133835/24-0620-140356/save_traj.npy # halfcheetah medium 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/halfcheetah-medium-replay-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143728/24-0620-140358/save_traj.npy # halfcheetah replay 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/halfcheetah-medium-replay-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143732/24-0620-140400/save_traj.npy # halfcheetah replay 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/halfcheetah-medium-replay-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143736/24-0620-140401/save_traj.npy # halfcheetah replay 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/halfcheetah-medium-expert-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-151455/24-0620-140403/save_traj.npy # halfcheetah expert 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/halfcheetah-medium-expert-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-151459/24-0620-140405/save_traj.npy # halfcheetah expert 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/halfcheetah-medium-expert-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-152153/24-0620-140407/save_traj.npy # halfcheetah expert 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/hopper-medium-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95_cond_length_2/diff_date_24-0606-225735_upervf_date_24-0611-161744/24-0620-140409/save_traj.npy # hopper medium 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/hopper-medium-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95_cond_length_5/diff_date_24-0606-225733_upervf_date_24-0611-161752/24-0620-140411/save_traj.npy # hopper medium 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/hopper-medium-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95_cond_length_10/diff_date_24-0606-225731_upervf_date_24-0611-161800/24-0620-140413/save_traj.npy # hopper medium 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/hopper-medium-replay-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143703/24-0620-140415/save_traj.npy # hopper medium replay 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/hopper-medium-replay-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143707/24-0620-140417/save_traj.npy # hopper medium replay 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/hopper-medium-replay-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143711/24-0620-140419/save_traj.npy # hopper medium replay 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/hopper-medium-expert-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0613-133751/24-0620-140421/save_traj.npy # hopper medium expert 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/hopper-medium-expert-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0613-133755/24-0620-140423/save_traj.npy # hopper medium expert 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/hopper-medium-expert-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0613-133759/24-0620-140425/save_traj.npy # hopper medium expert 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_4_head_2/diff_date_24-0618-143025_upervf_date_24-0623-184606/24-0624-113038/save_traj.npy # walker2d medium 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_4_head_2/diff_date_24-0618-143025_upervf_date_24-0623-184610/24-0624-113040/save_traj.npy # walker2d medium 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_4_head_2/diff_date_24-0618-143025_upervf_date_24-0623-184615/24-0624-113042/save_traj.npy # walker2d medium 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-replay-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143740/24-0620-140433/save_traj.npy # walker2d medium replay 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-replay-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143744/24-0620-140435/save_traj.npy # walker2d medium replay 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-replay-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143748/24-0620-140437/save_traj.npy # walker2d medium replay 10
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-expert-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_4_head_2/diff_date_24-0608-210615_upervf_date_24-0623-184619/24-0624-113044/save_traj.npy # walker2d medium expert 2
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-expert-v2/diff_horizon_20_cond_length_5/uper_vf_er_0.95cond_length5_layer_4_head_2/diff_date_24-0608-210615_upervf_date_24-0623-184623/24-0624-113046/save_traj.npy # walker2d medium expert 5
                          /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-expert-v2/diff_horizon_20_cond_length_10/uper_vf_er_0.95cond_length10_layer_4_head_2/diff_date_24-0608-210615_upervf_date_24-0623-184627/24-0624-113048/save_traj.npy # walker2d medium expert 10
                          )
upervf_path=(/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-v2/er_0.95cond_length2_layer_3_head_1/24-0613-133827/dt_checkpoint.pt # halfcheetah medium 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-v2/er_0.95cond_length5_layer_3_head_1/24-0613-133831/dt_checkpoint.pt # halfcheetah medium 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-v2/er_0.95cond_length10_layer_3_head_1/24-0613-133835/dt_checkpoint.pt # halfcheetah medium 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-143728/uper_value_func_checkpoint.pt # halfcheetah medium replay 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-143732/uper_value_func_checkpoint.pt # halfcheetah medium replay 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-143736/uper_value_func_checkpoint.pt # halfcheetah medium replay 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-151455/uper_value_func_checkpoint.pt # halfcheetah medium expert 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0618-151459/uper_value_func_checkpoint.pt # halfcheetah medium expert 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0618-152153/uper_value_func_checkpoint.pt # halfcheetah medium expert 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_2_layer_3_head_1/24-0611-161744/uper_value_func_checkpoint.pt # hopper medium 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_5_layer_3_head_1/24-0611-161752/uper_value_func_checkpoint.pt # hopper medium 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_10_layer_3_head_1/24-0611-161800/uper_value_func_checkpoint.pt # hopper medium 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-143703/uper_value_func_checkpoint.pt # hopper medium replay 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-143707/uper_value_func_checkpoint.pt # hopper medium replay 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-143711/uper_value_func_checkpoint.pt # hopper medium replay 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0613-133751/dt_checkpoint.pt # hopper medium expert 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0613-133755/dt_checkpoint.pt # hopper medium expert 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0613-133759/dt_checkpoint.pt # hopper medium expert 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func_0623/halfcond_transformer_noreward/walker2d-medium-v2/er_0.95cond_length2_layer_4_head_2/24-0623-184606/uper_value_func_checkpoint.pt # walker2d medium 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func_0623/halfcond_transformer_noreward/walker2d-medium-v2/er_0.95cond_length5_layer_4_head_2/24-0623-184610/uper_value_func_checkpoint.pt # walker2d medium 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func_0623/halfcond_transformer_noreward/walker2d-medium-v2/er_0.95cond_length10_layer_4_head_2/24-0623-184615/uper_value_func_checkpoint.pt # walker2d medium 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func_0623/halfcond_transformer_noreward/walker2d-medium-replay-v2/er_0.95cond_length2_layer_4_head_2/24-0624-111552/uper_value_func_checkpoint.pt # walker2d medium replay 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func_0623/halfcond_transformer_noreward/walker2d-medium-replay-v2/er_0.95cond_length5_layer_4_head_2/24-0624-111556/uper_value_func_checkpoint.pt # walker2d medium replay 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func_0623/halfcond_transformer_noreward/walker2d-medium-replay-v2/er_0.95cond_length10_layer_4_head_2/24-0624-111600/uper_value_func_checkpoint.pt # walker2d medium replay 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func_0623/halfcond_transformer_noreward/walker2d-medium-expert-v2/er_0.95cond_length2_layer_4_head_2/24-0623-184631/uper_value_func_checkpoint.pt # walker2d medium expert 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func_0623/halfcond_transformer_noreward/walker2d-medium-expert-v2/er_0.95cond_length5_layer_4_head_2/24-0623-184635/uper_value_func_checkpoint.pt # walker2d medium expert 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func_0623/halfcond_transformer_noreward/walker2d-medium-expert-v2/er_0.95cond_length10_layer_4_head_2/24-0623-184639/uper_value_func_checkpoint.pt # walker2d medium expert 10
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
    for ((j=0;j<3;j++)) # cond length 3
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

